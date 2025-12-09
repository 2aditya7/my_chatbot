# services/gemini_service.py
import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai.errors import APIError
from fastapi import HTTPException
from .ba_persona import ELABORATED_BA_ANALYSIS_PROMPT

GEMINI_MODEL = "gemini-2.5-flash"

def generate_ba_analysis_gemini(business_req: str) -> str:
    full_prompt = ELABORATED_BA_ANALYSIS_PROMPT.format(business_req=business_req)
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt
        )
        return response.text
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Gemini APIError: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")
