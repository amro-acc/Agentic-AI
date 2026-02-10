import warnings
import os
from crewai import Agent, Task, Crew, LLM
from utils.get_groq_api_key import get_groq_api_key
from utils.get_serper_api_key import get_serper_api_key
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel
import json
from pprint import pprint


def main():
    warnings.filterwarnings('ignore')

    # ========================
    # Patch SerperDevTool to accept dict safely
    # ========================
    # Keep a reference to the original method
    _original_run = SerperDevTool.run

    def safe_run(self, search_query):
        # Ensure the argument is always a plain string
        if isinstance(search_query, dict):
            search_query = search_query.get("description", str(search_query))
        # Call the *original* method safely
        return _original_run(self, search_query)

    SerperDevTool.run = safe_run
    # ========================

    # Use Groq (OpenAI-compatible API)
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
    os.environ["GROQ_API_KEY"] = get_groq_api_key()
    os.environ["SERPER_API_KEY"] = get_serper_api_key()

    llm = LLM(model="groq/llama-3.3-70b-versatile")  # this model supports 12K tokens per minute

    # Initialize the tools
    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()

    # Agent 1: Venue Coordinator
    venue_coordinator = Agent(
        role="Venue Coordinator",
        goal="Identify and book an appropriate venue based on event requirements",
        tools=[search_tool, scrape_tool],
        verbose=True,
        backstory=(
            "With a keen sense of space and understanding of event logistics, "
            "you excel at finding and securing the perfect venue that fits "
            "the event's theme, size, and budget constraints."
        ),
        llm=llm
    )

    # Agent 2: Logistics Manager
    logistics_manager = Agent(
        role="Logistics Manager",
        goal=(
            "Manage all logistics for the event including catering and equipment"
        ),
        tools=[search_tool, scrape_tool],
        verbose=True,
        backstory=(
            "Organized and detail-oriented, you ensure that every logistical "
            "aspect of the event from catering to equipment setup is flawlessly executed."
        ),
        llm=llm
    )

    # Agent 3: Marketing and Communications Agent
    marketing_communications_agent = Agent(
        role="Marketing and Communications Agent",
        goal="Effectively market the event and communicate with participants",
        tools=[search_tool, scrape_tool],
        verbose=True,
        backstory=(
            "Creative and communicative, you craft compelling messages and engage "
            "with potential attendees to maximize event exposure and participation."
        ),
        llm=llm
    )

    class VenueDetails(BaseModel):
        name: str
        address: str
        capacity: int
        booking_status: str

    venue_task = Task(
        description=(
            "Find a venue in {event_city} that meets criteria for {event_topic}. "
            "Use the search tool with a plain text query like "
            "'best venues in San Francisco for tech conference'."
        ),
        expected_output=(
            "All the details of a specifically chosen venue you found to accommodate the event. "
            "Return valid JSON only, like: {\"website_url\": \"...\"}. "
            "Return a JSON object with numeric 'capacity' (integer only) and 'booking_status' "
            "(string: 'available' or 'unavailable')."
        ),
        human_input=True,
        output_json=VenueDetails,
        output_file="venue_details.json",
        agent=venue_coordinator,
    )

    logistics_task = Task(
        description=(
            "Coordinate catering and equipment for an event with "
            "{expected_participants} participants on {tentative_date}."
        ),
        expected_output=(
            "Confirmation of all logistics arrangements including catering and equipment setup."
        ),
        human_input=True,
        async_execution=False,
        agent=logistics_manager,
    )

    marketing_task = Task(
        description=(
            "Promote the {event_topic} aiming to engage at least "
            "{expected_participants} potential attendees."
        ),
        expected_output=(
            "Report on marketing activities and attendee engagement formatted as markdown."
        ),
        async_execution=True,
        output_file="marketing_report.md",
        agent=marketing_communications_agent,
    )

    # Define the crew with agents and tasks
    event_management_crew = Crew(
        agents=[
            venue_coordinator,
            logistics_manager,
            marketing_communications_agent,
        ],
        tasks=[venue_task, logistics_task, marketing_task],
        verbose=True,
    )

    event_details = {
        "event_topic": "Tech Innovation Conference",
        "event_description": (
            "A gathering of tech innovators and industry leaders "
            "to explore future technologies."
        ),
        "event_city": "San Francisco",
        "tentative_date": "2024-09-15",
        "expected_participants": 500,
        "budget": 20000,
        "venue_type": "Conference Hall",
    }

    result = event_management_crew.kickoff(inputs=event_details)

    # Load and print the saved venue details
    with open("venue_details.json") as f:
        data = json.load(f)
    pprint(data)

    # Display marketing report
    with open("marketing_report.md", "r") as f:
        print(f.read())


if __name__ == "__main__":
    main()
