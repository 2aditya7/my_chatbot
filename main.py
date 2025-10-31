import os
import sys
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI API KEY NOT FOUND")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=api_key)

PYTHON_TUTOR_PROMPT = "You are a very helpful and encouraging tutor but a little sarcasm with an interesting emoji at the end . Keep the explanations simple and use analogies to explain the complex topics."

print("Gemini AI Chatbot Initialized.")

user_prompt = input("What would you like to ask? Type your question here : \n>")

print("\n thinking")


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents= user_prompt,
    config=genai.types.GenerateContentConfig(
        system_instruction=PYTHON_TUTOR_PROMPT
    )
)
print("\n AI Response:")
print(response.text)