# pylint: disable=C0114
import warnings
import os
from datetime import datetime
from crewai import Agent, Task, Crew
#from IPython.display import Markdown
#from utils.get_openai_api_key import get_openai_api_key

# pylint: disable=C0114
def main():

    warnings.filterwarnings('ignore')

    # Commented out to use Ollama locally so that no OpenAI API rate limits hit
    # openai_api_key = get_openai_api_key()
    # os.environ["OPENAI_API_KEY"] = openai_api_key
    # os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

    # Use Ollama(Own llama) locally instead of OpenAI
    os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"  # dummy value, required by CrewAI
    os.environ["OPENAI_MODEL_NAME"] = "llama3.2:3b"



    planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    allow_delegation=False,
    verbose=True
    )

    writer = Agent(
      role="Content Writer",
      goal="Write insightful and factually accurate "
          "opinion piece about the topic: {topic}",
      backstory="You're working on a writing "
                "a new opinion piece about the topic: {topic}. "
                "You base your writing on the work of "
                "the Content Planner, who provides an outline "
                "and relevant context about the topic. "
                "You follow the main objectives and "
                "direction of the outline, "
                "as provide by the Content Planner. "
                "You also provide objective and impartial insights "
                "and back them up with information "
                "provide by the Content Planner. "
                "You acknowledge in your opinion piece "
                "when your statements are opinions "
                "as opposed to objective statements.",
      allow_delegation=False,
      verbose=True
    )

    editor = Agent(
      role="Editor",
      goal="Edit a given blog post to align with "
          "the writing style of the organization. ",
      backstory="You are an editor who receives a blog post "
                "from the Content Writer. "
                "Your goal is to review the blog post "
                "to ensure that it follows journalistic best practices,"
                "provides balanced viewpoints "
                "when providing opinions or assertions, "
                "and also avoids major controversial topics "
                "or opinions when possible.",
      allow_delegation=False,
      verbose=True
    )

    plan = Task(
      description=(
          "1. Prioritize the latest trends, key players, "
              "and noteworthy news on {topic}.\n"
          "2. Identify the target audience, considering "
              "their interests and pain points.\n"
          "3. Develop a detailed content outline including "
              "an introduction, key points, and a call to action.\n"
          "4. Include SEO keywords and relevant data or sources."
      ),
      expected_output="A comprehensive content plan document "
          "with an outline, audience analysis, "
          "SEO keywords, and resources.",
      agent=planner,
    )

    write = Task(
      description=(
          "1. Use the content plan to craft a compelling "
              "blog post on {topic}.\n"
          "2. Incorporate SEO keywords naturally.\n"
      "3. Sections/Subtitles are properly named "
              "in an engaging manner.\n"
          "4. Ensure the post is structured with an "
              "engaging introduction, insightful body, "
              "and a summarizing conclusion.\n"
          "5. Proofread for grammatical errors and "
              "alignment with the brand's voice.\n"
      ),
      expected_output="A well-written blog post "
          "in markdown format, ready for publication, "
          "each section should have 2 or 3 paragraphs.",
      agent=writer,
    )

    edit = Task(
      description=("Proofread the given blog post for "
                  "grammatical errors and "
                  "alignment with the brand's voice."),
      expected_output="A well-written blog post in markdown format, "
                      "ready for publication, "
                      "each section should have 2 or 3 paragraphs.",
      agent=editor
    )

    crew = Crew(
      agents=[planner, writer, editor],
      tasks=[plan, write, edit],
      verbose=True
    )

    result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})
  #  Markdown(result)

    # Safely extract the final text regardless of CrewAI version
    output_text = None
    if hasattr(result, "final_output"):
        output_text = result.final_output
    elif hasattr(result, "output_text"):
        output_text = result.output_text
    elif hasattr(result, "output"):
        output_text = result.output
    elif isinstance(result, dict):
        output_text = result.get("final_output") or result.get("output") or str(result)
    else:
        output_text = str(result)

    # --- Print nicely formatted output ---
    print("\n" + "="*80)
    print("🧠 FINAL OUTPUT:")
    print("="*80 + "\n")
    print(output_text)
    print("\n" + "="*80)

    # --- Save output to Markdown file ---
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Optional: create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"crew_output_{timestamp}.md")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"✅ Output saved to: {output_file}")

if __name__ == "__main__":
    main()
