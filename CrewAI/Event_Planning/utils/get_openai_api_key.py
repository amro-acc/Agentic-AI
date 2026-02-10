# pylint: disable=C0114
import os
from dotenv import load_dotenv

def get_openai_api_key():
    """
    Loads OPENAI_API_KEY from environment or .env file.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or .env file")
    return api_key
