import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    sys.exit("GEMINI API KEY NOT FOUND")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
genai.configure(api_key=api_key)

PYTHON_TUTOR_PROMPT = "You are a very helpful and encouraging tutor but a little sarcasm with an interesting emoji at the end . Keep the explanations simple and use analogies to explain the complex topics."

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=PYTHON_TUTOR_PROMPT
)

chat = model.start_chat()

print("🤖 Gemini Chatbot Initialized with Persona. Type 'quit' or 'exit' to stop.")
print("----------------------------------------------------------------------")

while True:
    user_prompt = input("You:")

    if user_prompt.lower() in ["quit","exit"]:
        print("\n Goodbye!")
        break

    if not user_prompt.strip():
        print("Please enter a non-empty message")
        continue

    print("\n Thinking...")

    response = chat.send_message(user_prompt)

    print("\n AI Response:")
    print(response.text)
    print("----------------------------------------------------------------------")