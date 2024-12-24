import openai
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to analyze and optimize prompts
def optimize_prompt(prompt):
    # Predefined optimizations
    optimizations = [
        {"pattern": "In order to", "suggestion": "To"},
        {"pattern": "Due to the fact that", "suggestion": "Because"},
        {"pattern": "It is important to note that", "suggestion": "Note that"},
        {"pattern": "For the purpose of", "suggestion": "For"},
    ]
    
    optimized_prompt = prompt
    suggestions = []

    for opt in optimizations:
        if opt["pattern"] in prompt:
            optimized_prompt = optimized_prompt.replace(opt["pattern"], opt["suggestion"])
            suggestions.append(f"Replace '{opt['pattern']}' with '{opt['suggestion']}'.")

    return optimized_prompt, suggestions

# Function to track OpenAI usage
def track_openai_usage(prompt, model="gpt-3.5-turbo"):
    try:
        # Optimize the prompt
        optimized_prompt, suggestions = optimize_prompt(prompt)

        # Make API request
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": optimized_prompt}
            ]
        )

        # Extract token usage and response details
        total_tokens = response["usage"]["total_tokens"]
        prompt_tokens = response["usage"]["prompt_tokens"]
        completion_tokens = response["usage"]["completion_tokens"]
        cost = calculate_cost(total_tokens, model)

        return {
            "response": response["choices"][0]["message"]["content"],
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "suggestions": suggestions
        }
    except Exception as e:
        return {"error": str(e)}

# Function to calculate cost
def calculate_cost(total_tokens, model):
    pricing = {"gpt-3.5-turbo": 0.002, "gpt-4": 0.03}  # Per 1k tokens
    cost_per_token = pricing.get(model, 0.001)
    return (total_tokens / 1000) * cost_per_token

# Streamlit App
st.title("LLM Usage Insights with Cost Optimization")

# Sidebar for user input
st.sidebar.header("LLM Insights")
prompt = st.sidebar.text_area("Enter your LLM prompt:")

# Display insights upon user action
if st.sidebar.button("Optimize and Track Usage"):
    if prompt:
        result = track_openai_usage(prompt)
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("Optimized Prompt Suggestions")
            st.write("Suggestions for optimization:")
            for suggestion in result["suggestions"]:
                st.write(f"- {suggestion}")

            st.subheader("Optimized Response")
            st.write(result["response"])

            st.subheader("Usage Details")
            st.write(f"**Total Tokens Used:** {result['total_tokens']}")
            st.write(f"**Prompt Tokens:** {result['prompt_tokens']}")
            st.write(f"**Completion Tokens:** {result['completion_tokens']}")
            st.write(f"**Cost of Request:** ${result['cost']:.4f}")
    else:
        st.warning("Please enter a prompt.")
