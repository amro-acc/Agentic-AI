import os
from crewai import Agent, Task, Crew

# Configure CrewAI to use local Ollama
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"  # dummy
os.environ["OPENAI_MODEL_NAME"] = "llama3.2:3b"

# Define a simple agent
haiku_agent = Agent(
    role="Poet",
    goal="Write a haiku about DevOps and teamwork.",
    backstory="A reflective poet who finds beauty in automation.",
    verbose=True
)

# Define a single task
haiku_task = Task(
    description="Write a short haiku about DevOps culture.",
    expected_output="A 3-line haiku with a meaningful message.",
    agent=haiku_agent
)

# Run the crew
crew = Crew(agents=[haiku_agent], tasks=[haiku_task], verbose=True)
result = crew.kickoff()

print("\n=== Haiku Output ===\n")
print(result)
print("\n====================\n")
