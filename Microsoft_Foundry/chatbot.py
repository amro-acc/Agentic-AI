import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI credentials from environment variables
AZURE_AI_FOUNDRY_PROJECT_ENDPOINT = os.getenv("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT")
AZURE_AI_FOUNDRY_MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_AI_FOUNDRY_MODEL_DEPLOYMENT_NAME")

print(f"Using Azure AI Foundry Project Endpoint: {AZURE_AI_FOUNDRY_PROJECT_ENDPOINT}")
print(f"Using Azure AI Foundry Model Deployment Name: {AZURE_AI_FOUNDRY_MODEL_DEPLOYMENT_NAME}\n")

# Initialize Azure AI Projects client
credential = DefaultAzureCredential()

project_client = AIProjectClient(
    endpoint=AZURE_AI_FOUNDRY_PROJECT_ENDPOINT,
    credential=credential
)

openai_client = project_client.get_openai_client()

# --- Interactive Loop Section ---

# 1. Initial greeting
print("AI: Hello! How can I help you today? (Type 'quit' to exit)")

while True:
    # 2. Capture user input from the command line
    user_input = input("\nYou: ")
    
    # 3. Check for exit condition
    if user_input.strip().lower() == "quit":
        print("AI: Goodbye!")
        break
        
    # 4. Skip empty inputs
    if not user_input.strip():
        continue
        
    try:
        # 5. Generate and print the response
        response = openai_client.responses.create(
            model=AZURE_AI_FOUNDRY_MODEL_DEPLOYMENT_NAME,
            input=user_input,
        )
        print(f"AI: {response.output_text}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
