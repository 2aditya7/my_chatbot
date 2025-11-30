# services/gemini_service.py
import os
from google import genai
from google.genai.errors import APIError
from .ba_persona import BA_ANALYSIS_PROMPT
from fastapi import HTTPException # Needed to raise 500 errors

GEMINI_MODEL = "gemini-2.5-flash" 

def generate_ba_analysis_gemini(business_req: str) -> str:
    """
    Connects to the Gemini API to generate BA requirements (non-streaming, simple text).
    """
    
    # 1. Prepare the full prompt using the centralized BA persona
    full_prompt = BA_ANALYSIS_PROMPT.format(business_req=business_req)

    try:
        # 2. Client automatically picks up GEMINI_API_KEY from environment variables (.env file)
        client = genai.Client() 
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt
        )
        
        return response.text
        
    except APIError as e:
        # Use FastAPI's HTTPException to return a proper 500 error to the client
        raise HTTPException(status_code=500, detail=f"CRITICAL ERROR: Failed to connect to Gemini API. Details: {e}")
    except Exception as e:
        # Handle other potential errors (like missing API key)
        raise HTTPException(status_code=500, detail=f"CRITICAL ERROR: General error in Gemini connection: {e}")

# NOTE: This is a synchronous (non-streaming) function for now, 
# which is why it's marked as "not yet implemented" in the main.py router.