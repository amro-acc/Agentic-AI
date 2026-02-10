from langchain_ollama import OllamaLLM

o = OllamaLLM(model="llama3.2:3b")
print(o.invoke("Say hello in one word"))