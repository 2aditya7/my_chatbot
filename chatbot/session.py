from .persona import PYTHON_TUTOR_PROMPT

def create_chat(genai):
    """Creates and returns a Gemini chat session with memory."""
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=PYTHON_TUTOR_PROMPT
    )
    return model.start_chat()

def send_message(chat, user_input):
    """Sends a message and returns the AI's text response."""
    response = chat.send_message(user_input)
    return response.text
