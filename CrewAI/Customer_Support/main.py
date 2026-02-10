"""Run a CrewAI customer-support demo.

This script configures a local model (Ollama) for CrewAI, constructs a support
agent and a support-quality-assurance agent, defines tasks that use a website
scraping tool, runs a Crew to resolve a customer inquiry, prints the final
output, and saves it as a timestamped Markdown file under the `outputs/`
directory.

Usage:
    python main.py

Notes:
- The script sets OPENAI_API_BASE, OPENAI_API_KEY (dummy), and OPENAI_MODEL_NAME
  to point CrewAI at a locally hosted Ollama instance.
- Side effects: environment variables are set, output is written to disk, and
  the Crew is executed which may make network calls.
"""

import warnings
import os
from datetime import datetime
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool


def main():
    
    """Run the CrewAI workflow to handle a customer inquiry.

    This function:
    - Silences warnings.
    - Configures environment variables to use a local Ollama model.
    - Instantiates two Agents (support and QA), corresponding Tasks, and a Crew.
    - Starts the Crew with sample inputs, prints the final output, and saves it
      to a timestamped Markdown file in `outputs/`.

    Returns:
        None

    Side effects:
        - Sets environment variables used by CrewAI.
        - Writes a Markdown file containing the final output.
        - Prints status and final output to stdout.
    """

    warnings.filterwarnings('ignore')

    # Use Ollama(Own llama) locally instead of OpenAI
    os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"  # dummy value, required by CrewAI
    os.environ["OPENAI_MODEL_NAME"] = "llama3.2:3b"

    support_agent = Agent(
        role="Senior Support Representative",
        goal="Be the most friendly and helpful "
            "support representative in your team",
        backstory=(
            "You work at crewAI (https://crewai.com) and "
            " are now working on providing "
            "support to {customer}, a super important customer "
            " for your company."
            "You need to make sure that you provide the best support!"
            "Make sure to provide full complete answers, "
            " and make no assumptions."
        ),
        allow_delegation=False,
        verbose=True
    )

    support_quality_assurance_agent = Agent(
        role="Support Quality Assurance Specialist",
        goal="Get recognition for providing the "
        "best support quality assurance in your team",
        backstory=(
            "You work at crewAI (https://crewai.com) and "
            "are now working with your team "
            "on a request from {customer} ensuring that "
            "the support representative is "
            "providing the best support possible.\n"
            "You need to make sure that the support representative "
            "is providing full"
            "complete answers, and make no assumptions."
        ),
        verbose=True
    )

    docs_scrape_tool = ScrapeWebsiteTool(
        website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/",
        name="Read CrewAI Docs",
        description=(
            "Reads and summarizes content from the CrewAI documentation page "
            "on creating and kicking off a crew."
        )
    )

    inquiry_resolution = Task(
        description=(
            "{customer} just reached out with a super important ask:\n"
            "{inquiry}\n\n"
            "{person} from {customer} is the one that reached out. "
            "Make sure to use everything you know "
            "to provide the best support possible."
            "You must strive to provide a complete "
            "and accurate response to the customer's inquiry."
            "Use the 'Read CrewAI Docs' tool to review the documentation "
            "before writing your final answer."
            "You can use the provided ScrapeWebsiteTool **only once** "
            "to read the documentation and extract the relevant answer. "
            "Do not call the tool repeatedly."
        ),
        expected_output=(
            "A detailed, informative response to the "
            "customer's inquiry that addresses "
            "all aspects of their question.\n"
            "The response should include references "
            "to everything you used to find the answer, "
            "including external data or solutions. "
            "Ensure the answer is complete, "
            "leaving no questions unanswered, and maintain a helpful and friendly "
            "tone throughout."
        ),
        tools=[docs_scrape_tool],
        agent=support_agent,
    )
    
    quality_assurance_review = Task(
        description=(
            "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
            "Ensure that the answer is comprehensive, accurate, and adheres to the "
            "high-quality standards expected for customer support.\n"
            "Verify that all parts of the customer's inquiry "
            "have been addressed "
            "thoroughly, with a helpful and friendly tone.\n"
            "Check for references and sources used to "
            " find the information, "
            "ensuring the response is well-supported and "
            "leaves no questions unanswered."
        ),
        expected_output=(
            "A final, detailed, and informative response "
            "ready to be sent to the customer.\n"
            "This response should fully address the "
            "customer's inquiry, incorporating all "
            "relevant feedback and improvements.\n"
            "Don't be too formal, we are a chill and cool company "
            "but maintain a professional and friendly tone throughout."
        ),
        agent=support_quality_assurance_agent,
    )

    crew = Crew(
      agents=[support_agent, support_quality_assurance_agent],
      tasks=[inquiry_resolution, quality_assurance_review],
      verbose=True,
      memory=False     # Memory not working with Ollama currently
    )

    inputs = {
        "customer": "DeepLearningAI",
        "person": "Andrew Ng",
        "inquiry": "I need help with setting up a Crew "
                   "and kicking it off, specifically "
                   "how can I add memory to my crew? "
                   "Can you provide guidance?"
    }
    result = crew.kickoff(inputs=inputs)

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
