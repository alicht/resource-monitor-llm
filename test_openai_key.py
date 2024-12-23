import openai
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Retrieve API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: No API key found. Ensure OPENAI_API_KEY is set in your .env file.")
else:
    print("Loaded API Key:", api_key[:10] + "..." + api_key[-5:])

# Test OpenAI API
try:
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    print("Response:", response.choices[0].message["content"].strip())
except openai.error.AuthenticationError as e:
    print("Authentication Error:", e)
except Exception as e:
    print("Error:", e)
