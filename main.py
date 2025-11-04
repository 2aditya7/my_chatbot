from chatbot.config import setup_gemini
from chatbot.session import create_chat, send_message

def main():
    genai = setup_gemini()
    chat = create_chat(genai)

    print("🤖 Gemini Chatbot Initialized. Type 'quit' or 'exit' to stop.")
    print("----------------------------------------------------------------------")

    while True:
        user_prompt = input("You: ")

        if user_prompt.lower() in ["quit", "exit"]:
            print("\n Goodbye!")
            break

        if not user_prompt.strip():
            print("Please enter a non-empty message")
            continue

        print("\n Thinking...")

        response = send_message(chat, user_prompt)
        print("\n AI Response:")
        print(response)
        print("----------------------------------------------------------------------")

if __name__ == "__main__":
    main()
