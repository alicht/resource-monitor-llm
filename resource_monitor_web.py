import os
import time
import csv
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import openai
from azure.identity import DefaultAzureCredential
from azure.mgmt.monitor import MonitorManagementClient

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Azure credentials setup
credential = DefaultAzureCredential()
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")

# File paths
llm_log_file = "llm_usage_logs.csv"

# Ensure log file exists
if not os.path.exists(llm_log_file):
    with open(llm_log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Total Tokens", "Prompt Tokens", "Cost"])

# Function to track OpenAI usage
def track_openai_usage(prompt, model="gpt-3.5-turbo"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        total_tokens = response["usage"]["total_tokens"]
        prompt_tokens = response["usage"]["prompt_tokens"]
        completion_tokens = response["usage"]["completion_tokens"]
        cost = calculate_cost(total_tokens, model)

        # Log usage
        with open(llm_log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                total_tokens, prompt_tokens, cost
            ])

        return {
            "response": response["choices"][0]["message"]["content"],
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost
        }
    except Exception as e:
        return {"error": str(e)}

# Function to calculate cost
def calculate_cost(total_tokens, model):
    pricing = {"gpt-3.5-turbo": 0.002, "gpt-4": 0.03}  # Per 1k tokens
    cost_per_token = pricing.get(model, 0.001)
    return (total_tokens / 1000) * cost_per_token

# Function to fetch Azure resource metrics
def fetch_azure_metrics():
    try:
        monitor_client = MonitorManagementClient(credential, subscription_id)
        metrics_data = monitor_client.metrics.list(
            resource_id=f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/virtualMachines/",
            timespan="PT1H",
            interval="PT1M",
            metricnames="Percentage CPU, Network In, Network Out",
            aggregation="Average"
        )

        metrics = {}
        for item in metrics_data.value:
            metrics[item.name.value] = [data.average for data in item.timeseries[0].data if data.average is not None]

        return metrics
    except Exception as e:
        return {"error": str(e)}

# Streamlit App
st.title("🌟 Resource Monitor with Predictive Insights")
st.markdown("This app provides **real-time monitoring**, **predictive analytics**, and **LLM usage insights** in a unified dashboard.")

# Sidebar for user input
st.sidebar.header("LLM Insights")
prompt = st.sidebar.text_area("Enter your LLM prompt:")
if st.sidebar.button("Track Usage"):
    if prompt:
        result = track_openai_usage(prompt)
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("🤖 OpenAI API Response")
            st.write(result["response"])

            st.subheader("📊 Usage Details")
            st.metric("Total Tokens Used", result["total_tokens"])
            st.metric("Prompt Tokens", result["prompt_tokens"])
            st.metric("Completion Tokens", result["completion_tokens"])
            st.metric("Cost of Request", f"${result['cost']:.4f}")

# Historical Logs and Trends
st.sidebar.header("Historical LLM Logs")
if st.sidebar.button("View Historical LLM Usage"):
    try:
        df = pd.read_csv(llm_log_file, header=0)

        st.subheader("🗂️ Historical LLM Usage Logs")
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 LLM Usage Trends")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(df["Total Tokens"], label="Total Tokens", marker="o", linestyle="--", color="blue")
        ax[0].set_title("Token Usage Over Time")
        ax[0].set_xlabel("Requests")
        ax[0].set_ylabel("Tokens")
        ax[0].legend()

        ax[1].plot(df["Cost"], label="Cost", marker="o", linestyle="--", color="green")
        ax[1].set_title("Cost Over Time")
        ax[1].set_xlabel("Requests")
        ax[1].set_ylabel("Cost ($)")
        ax[1].legend()

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading historical data: {e}")

# Predict Future Token Usage and Costs
st.sidebar.header("Predict Future Costs")
future_token_usage = st.sidebar.number_input("Enter future token usage:", min_value=1, step=1)
if st.sidebar.button("Predict Cost"):
    try:
        df = pd.read_csv(llm_log_file, header=0)
        df = df.dropna()  # Remove rows with missing values

        df = df[df["Total Tokens"].apply(lambda x: str(x).isdigit())]

        token_usage = df["Total Tokens"].astype(float).values.reshape(-1, 1)
        costs = df["Cost"].astype(float).values

        model = LinearRegression()
        model.fit(token_usage, costs)

        predicted_cost = model.predict([[future_token_usage]])[0]

        st.subheader("🔮 Future Token Usage Prediction")
        st.metric("Predicted Cost", f"${predicted_cost:.4f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Azure Resource Metrics
st.sidebar.header("Azure Resource Metrics")
if st.sidebar.button("Fetch Azure Metrics"):
    metrics = fetch_azure_metrics()
    if "error" in metrics:
        st.error(metrics["error"])
    else:
        st.subheader("☁️ Azure Resource Metrics")
        for key, values in metrics.items():
            st.metric(key, ", ".join([str(round(v, 2)) for v in values]))
        st.success("Fetched Azure metrics successfully.")
