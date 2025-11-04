import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv

def setup_gemini():
    """Loads API key from .env and configures Gemini."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        sys.exit(" GEMINI API KEY NOT FOUND")

    genai.configure(api_key=api_key)
    return genai
