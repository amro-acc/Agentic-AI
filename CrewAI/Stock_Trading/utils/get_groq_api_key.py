# pylint: disable=C0114
import os
from dotenv import load_dotenv

def get_groq_api_key():
    """
    Loads GROQ_API_KEY from environment or .env file.
    """
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment or .env file")
    return groq_api_key
