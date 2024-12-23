# import psutil
import streamlit as st
import matplotlib.pyplot as plt
import os
import time
import random
import csv
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found. Please set OPENAI_API_KEY in your .env file.")
else:
    openai.api_key = api_key

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
        cost = calculate_cost(total_tokens, model)
        return {
            "response": response["choices"][0]["message"]["content"],
            "tokens": total_tokens,
            "cost": cost
        }
    except openai.error.AuthenticationError as e:
        return {"error": f"Authentication Error: {e}"}
    except Exception as e:
        return {"error": f"Error: {e}"}

def calculate_cost(total_tokens, model):
    pricing = {"gpt-3.5-turbo": 0.002, "gpt-4": 0.03}
    return (total_tokens / 1000) * pricing.get(model, 0.001)

# Function to check system resources
def check_resources():
    return {
        "CPU Usage (%)": psutil.cpu_percent(interval=0),
        "Memory Usage (%)": psutil.virtual_memory().percent,
        "GPU Utilization (%)": random.randint(0, 70),
        "GPU Memory Usage (%)": random.randint(0, 80)
    }

# Function to simulate task execution
def simulate_task(task_id, duration=5):
    with st.spinner(f"Running Task {task_id}..."):
        time.sleep(duration)
    st.success(f"Task {task_id} completed!")

# Initialize Streamlit app
st.title("Spaire: AI Workflow Optimization")
st.write("Monitor system resources, track LLM usage, and optimize workflows.")

# Sidebar Configuration
st.sidebar.header("Configuration")
update_interval = st.sidebar.slider("Update Interval (seconds)", 1, 10, 2)
task_count = st.sidebar.number_input("Number of Tasks", min_value=1, max_value=20, value=5)

# Sidebar for LLM Tracking
st.sidebar.header("LLM Insights")
prompt = st.sidebar.text_area("Enter your LLM prompt:")
if st.sidebar.button("Send Prompt"):
    if prompt:
        result = track_openai_usage(prompt)
        if "error" in result:
            st.error(result["error"])
        else:
            st.write("Response:", result["response"])
            st.write(f"Tokens Used: {result['tokens']}")
            st.write(f"Cost: ${result['cost']:.4f}")
    else:
        st.warning("Please enter a prompt.")

# Real-Time Resource Monitoring
st.subheader("Live Resource Usage")
resource_placeholder = st.empty()
chart_placeholder = st.empty()

if "cpu_data" not in st.session_state:
    st.session_state.cpu_data = []
    st.session_state.memory_data = []
    st.session_state.gpu_data = []
    st.session_state.gpu_memory_data = []

if st.button("Start Monitoring"):
    for _ in range(10):  # Adjust for duration
        resources = check_resources()
        st.session_state.cpu_data.append(resources["CPU Usage (%)"])
        st.session_state.memory_data.append(resources["Memory Usage (%)"])
        st.session_state.gpu_data.append(resources["GPU Utilization (%)"])
        st.session_state.gpu_memory_data.append(resources["GPU Memory Usage (%)"])

        resource_placeholder.write(resources)

        # Update Graph
        fig, ax = plt.subplots()
        ax.plot(st.session_state.cpu_data, label="CPU Usage (%)")
        ax.plot(st.session_state.memory_data, label="Memory Usage (%)")
        ax.plot(st.session_state.gpu_data, label="GPU Utilization (%)")
        ax.plot(st.session_state.gpu_memory_data, label="GPU Memory Usage (%)")
        ax.legend()
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time")
        ax.set_ylabel("Usage (%)")
        ax.set_title("System Resource Usage (Real-Time)")
        chart_placeholder.pyplot(fig)
        time.sleep(update_interval)

# Historical Logs
st.subheader("Historical Logs")
log_file = "resource_logs.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "CPU Usage (%)", "Memory Usage (%)", "GPU Utilization (%)", "GPU Memory Usage (%)"])

if st.button("View Resource Logs"):
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        st.dataframe(df)
    else:
        st.warning("No logs found.")
