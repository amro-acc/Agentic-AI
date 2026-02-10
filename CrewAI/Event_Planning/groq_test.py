from openai import OpenAI
import os
from utils.get_groq_api_key import get_groq_api_key

os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama-3.1-8b-instant"
os.environ["GROQ_API_KEY"] = get_groq_api_key()


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=get_groq_api_key()
)

resp = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hi from Groq!"}]
)
print(resp.choices[0].message.content)
