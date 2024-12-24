import psutil
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import csv
import os
import random
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# File paths
llm_log_file = "llm_usage_logs.csv"
resource_log_file = "resource_logs.csv"

# Function to track OpenAI usage
def track_openai_usage(prompt, model="gpt-3.5-turbo"):
    try:
        # Make API request
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract token usage and response details
        total_tokens = response["usage"]["total_tokens"]
        prompt_tokens = response["usage"]["prompt_tokens"]
        completion_tokens = response["usage"]["completion_tokens"]
        cost = calculate_cost(total_tokens, model)

        # Log usage
        with open(llm_log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                total_tokens, prompt_tokens, completion_tokens, cost
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

# Function to clean malformed CSV rows
def clean_csv(file_path, required_columns):
    clean_rows = []
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        headers = next(reader)
        for row in reader:
            if len(row) == required_columns:
                clean_rows.append(row)
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(clean_rows)

# Streamlit App
st.title("Spaire: Optimize Your AI Workflows")
st.markdown(
    """
    #### Welcome to Spaire! 🎯
    Your one-stop solution for **LLM usage insights**, **cost optimization**, and **resource monitoring**.
    """
)

# Sidebar for LLM Insights
st.sidebar.header("LLM Insights")
prompt = st.sidebar.text_area("Enter your LLM prompt:")

if st.sidebar.button("Track Usage"):
    if prompt:
        result = track_openai_usage(prompt)
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("OpenAI API Response")
            st.markdown(f"**Response:** {result['response']}")

            st.subheader("Usage Details")
            st.write(f"**Total Tokens Used:** {result['total_tokens']}")
            st.write(f"**Prompt Tokens:** {result['prompt_tokens']}")
            st.write(f"**Completion Tokens:** {result['completion_tokens']}")
            st.write(f"**Cost of Request:** ${result['cost']:.4f}")
    else:
        st.warning("Please enter a prompt.")

# Historical LLM Usage Logs
st.subheader("Historical LLM Usage Logs and Trends")
if os.path.exists(llm_log_file):
    clean_csv(llm_log_file, 4)
    df = pd.read_csv(llm_log_file, names=["Timestamp", "Total Tokens", "Prompt Tokens", "Completion Tokens", "Cost"])
    st.dataframe(df)

    # Visualization
    st.subheader("Trends Over Time")
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(df.index, df["Total Tokens"], label="Total Tokens", marker="o")
    ax[0].set_title("Total Token Usage")
    ax[0].set_xlabel("Requests")
    ax[0].set_ylabel("Tokens")
    ax[0].legend()

    ax[1].plot(df.index, df["Cost"], label="Cost", color="green", marker="o")
    ax[1].set_title("Cost Per Request")
    ax[1].set_xlabel("Requests")
    ax[1].set_ylabel("Cost (USD)")
    ax[1].legend()

    st.pyplot(fig)
else:
    st.warning("No historical LLM usage logs found.")

# Footer
st.markdown("---")
st.markdown("Powered by Spaire: Optimize your AI-specific workflows effortlessly.")
