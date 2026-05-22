import os
import sys
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from typing import List

# Ensure correct pathing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from vectornaut.config import get_client

class ParameterProposal(BaseModel):
    name: str
    value: float

class GeminiChatResponse(BaseModel):
    reply: str
    suggested_params: List[ParameterProposal] = []

def test_direct():
    client = get_client()
    system_instructions = (
        "Du bist der bionische Assistenz-Bot von Vectornaut. "
        "Bitte antworte auf Deutsch mit korrekten Umlauten (ä, ö, ü, ß). "
        "Schreibe eine kurze Erklärung zur Reibungsreduktion."
    )
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Erkläre mir das bitte.",
        config=dict(
            system_instruction=system_instructions,
            response_mime_type="application/json",
            response_schema=GeminiChatResponse,
        )
    )
    
    print("API Response parsed:")
    print("Reply:", response.parsed.reply)

if __name__ == "__main__":
    test_direct()
