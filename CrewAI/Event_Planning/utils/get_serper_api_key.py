# pylint: disable=C0114
import os
from dotenv import load_dotenv

def get_serper_api_key():
    """
    Loads SERPER_API_KEY from environment or .env file.
    """
    load_dotenv()
    serper_api_key = os.getenv("SERPER_API_KEY")

    if not serper_api_key:
        raise ValueError("SERPER_API_KEY not found in environment or .env file")
    return serper_api_key
