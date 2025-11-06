import streamlit as st
from chatbot.config import setup_gemini
from chatbot.session import create_chat, send_message

# Streamlit page setup
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Gemini Chatbot")

# Initialize chat session
if "chat" not in st.session_state:
    genai = setup_gemini()
    st.session_state.chat = create_chat(genai)
    st.session_state.history = []  # store messages

# Display chat history
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# User input
if user_input := st.chat_input("Type your message..."):
    st.session_state.history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_message(st.session_state.chat, user_input)
            st.markdown(response)

    st.session_state.history.append({"role": "assistant", "content": response})
