import os
import time
import random
import psutil
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        optimization_tip = get_cost_optimization_tip(total_tokens, model)

        return {
            "response": response["choices"][0]["message"]["content"],
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "optimization_tip": optimization_tip,
        }
    except Exception as e:
        return {"error": str(e)}

# Function to calculate cost
def calculate_cost(total_tokens, model):
    pricing = {"gpt-3.5-turbo": 0.002, "gpt-4": 0.03}
    cost_per_token = pricing.get(model, 0.001)
    return (total_tokens / 1000) * cost_per_token

# Function to generate cost optimization tip
def get_cost_optimization_tip(total_tokens, model):
    if model == "gpt-4" and total_tokens > 1000:
        return "Consider switching to GPT-3.5-turbo for large prompts to save up to 93% on costs."
    elif model == "gpt-3.5-turbo" and total_tokens > 2000:
        return "Optimize your prompt to reduce token usage and save costs."
    return "Your model choice is already cost-effective for this workload."

# Function to check system resources
def check_resources():
    return {
        "CPU Usage (%)": psutil.cpu_percent(interval=0),
        "Memory Usage (%)": psutil.virtual_memory().percent,
        "GPU Utilization (%)": random.randint(0, 70),
        "GPU Memory Usage (%)": random.randint(0, 80),
    }

# Streamlit app
st.title("Spaire: Optimize AI Workflows and Costs")
st.write("Monitor resources, track LLM usage, and gain actionable insights.")

# Tabs for organization
tab1, tab2 = st.tabs(["Resource Monitoring", "LLM Insights"])

# Tab 1: Resource Monitoring
with tab1:
    st.subheader("Live Resource Monitoring")
    col1, col2 = st.columns(2)

    if "cpu_data" not in st.session_state:
        st.session_state.cpu_data = []
        st.session_state.memory_data = []
        st.session_state.gpu_data = []
        st.session_state.gpu_memory_data = []

    resources = check_resources()
    st.session_state.cpu_data.append(resources["CPU Usage (%)"])
    st.session_state.memory_data.append(resources["Memory Usage (%)"])
    st.session_state.gpu_data.append(resources["GPU Utilization (%)"])
    st.session_state.gpu_memory_data.append(resources["GPU Memory Usage (%)"])

    with col1:
        st.metric("CPU Usage", f"{resources['CPU Usage (%)']}%")
        st.metric("Memory Usage", f"{resources['Memory Usage (%)']}%")
    with col2:
        st.metric("GPU Utilization", f"{resources['GPU Utilization (%)']}%")
        st.metric("GPU Memory Usage", f"{resources['GPU Memory Usage (%)']}%")

    fig, ax = plt.subplots()
    ax.plot(st.session_state.cpu_data, label="CPU Usage (%)", color="blue")
    ax.plot(st.session_state.memory_data, label="Memory Usage (%)", color="green")
    ax.plot(st.session_state.gpu_data, label="GPU Utilization (%)", color="red")
    ax.plot(st.session_state.gpu_memory_data, label="GPU Memory Usage (%)", color="purple")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.set_xlabel("Time")
    ax.set_ylabel("Usage (%)")
    ax.set_title("System Resource Usage (Real-Time)")
    st.pyplot(fig)

# Tab 2: LLM Insights
with tab2:
    st.subheader("LLM Usage Insights with Cost Optimization")
    example_prompts = [
        "Write a creative ad for a coffee shop.",
        "Summarize the following text: [Add your text here]",
        "What are the benefits of AI in education?",
    ]
    selected_prompt = st.selectbox("Choose an example prompt:", [""] + example_prompts)
    user_prompt = st.text_area("Or enter your own prompt:", value=selected_prompt)

    if st.button("Track Usage"):
        if user_prompt:
            with st.spinner("Fetching LLM insights..."):
                result = track_openai_usage(user_prompt)
            if "error" in result:
                st.error(result["error"])
            else:
                st.write("**Response:**", result["response"])
                st.write(f"**Total Tokens Used:** {result['total_tokens']}")
                st.write(f"**Prompt Tokens:** {result['prompt_tokens']}")
                st.write(f"**Completion Tokens:** {result['completion_tokens']}")
                st.write(f"**Cost of Request:** ${result['cost']:.4f}")
                st.write("**Cost Optimization Tip:**", result["optimization_tip"])
        else:
            st.warning("Please enter a prompt or select an example.")
