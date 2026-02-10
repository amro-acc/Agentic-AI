
import warnings
"""
Event Planning System using CrewAI Framework.
This script orchestrates an event planning system using multiple AI agents working together.
It handles venue selection, logistics management, and marketing for events.
The system uses three main agents:
1. Venue Coordinator - Finds and books appropriate venues
2. Logistics Manager - Handles catering and equipment arrangements
3. Marketing Communications Agent - Manages event promotion and attendee engagement
The script utilizes external APIs (OpenAI and Serper) and tools for web searching and scraping.
Functions:
    main(): Initializes and executes the event planning workflow
Required Environment Variables:
    - OPENAI_API_KEY: API key for OpenAI services
    - SERPER_API_KEY: API key for Serper search services
Output Files:
    - venue_details.json: Contains details of the selected venue
    - marketing_report.md: Marketing activities and engagement report
Dependencies:
    - crewai
    - pydantic
    - warnings
    - os
    - datetime
    - json
    - IPython
Example Usage:
    python main.py
Note:
    Requires valid API keys and necessary permissions for external services.
"""
import os
from crewai import Agent, Task, Crew
from utils.get_openai_api_key import get_openai_api_key
from utils.get_serper_api_key import get_serper_api_key
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel
import json
from pprint import pprint

def main():
    """
    Orchestrates an event planning system using CrewAI framework with multiple specialized agents.
    This function sets up and executes an event planning workflow with three main agents:
    - Venue Coordinator: Handles venue selection and booking
    - Logistics Manager: Manages catering and equipment
    - Marketing and Communications Agent: Handles event promotion
    The function performs the following steps:
    1. Sets up environment variables and API keys
    2. Initializes search and web scraping tools
    3. Creates specialized agents with defined roles and goals
    4. Defines tasks for each agent with expected outputs
    5. Creates a crew to manage the agents and their tasks
    6. Executes the event planning workflow with provided event details
    Returns:
        None. Outputs are written to:
        - venue_details.json: Contains details of the selected venue
        - marketing_report.md: Contains marketing activities report
    Required Environment Variables:
        - OPENAI_API_KEY: API key for OpenAI services
        - SERPER_API_KEY: API key for Serper dev tools
    Dependencies:
        - warnings
        - os
        - json
        - pprint
        - CrewAI framework classes (Agent, Task, Crew)
        - Custom tools (SerperDevTool, ScrapeWebsiteTool)
    """

    warnings.filterwarnings('ignore')

    # Use Ollama(Own llama) locally instead of OpenAI
    os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
    os.environ["OPENAI_MODEL_NAME"] = "llama3.2:3b"
    os.environ["OPENAI_API_KEY"] = get_openai_api_key()
    os.environ["SERPER_API_KEY"] = get_serper_api_key()
 

    # Initialize the tools
    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()

    # Agent 1: Venue Coordinator
    venue_coordinator = Agent(
        role="Venue Coordinator",
        goal="Identify and book an appropriate venue "
        "based on event requirements",
        tools=[search_tool, scrape_tool],
        verbose=True,
        backstory=(
            "With a keen sense of space and "
            "understanding of event logistics, "
            "you excel at finding and securing "
            "the perfect venue that fits the event's theme, "
            "size, and budget constraints."
        )
    )

    # Agent 2: Logistics Manager
    logistics_manager = Agent(
        role='Logistics Manager',
        goal=(
            "Manage all logistics for the event "
            "including catering and equipment"
        ),
        tools=[search_tool, scrape_tool],
        verbose=True,
        backstory=(
            "Organized and detail-oriented, "
            "you ensure that every logistical aspect of the event "
            "from catering to equipment setup "
            "is flawlessly executed to create a seamless experience."
        )
    )

    # Agent 3: Marketing and Communications Agent
    marketing_communications_agent = Agent(
        role="Marketing and Communications Agent",
        goal="Effectively market the event and "
            "communicate with participants",
        tools=[search_tool, scrape_tool],
        verbose=True,
        backstory=(
            "Creative and communicative, "
            "you craft compelling messages and "
            "engage with potential attendees "
            "to maximize event exposure and participation."
        )
    )

    class VenueDetails(BaseModel):
        name: str
        address: str
        capacity: int
        booking_status: str

    venue_task = Task(
        description="Find a venue in {event_city} "
                    "that meets criteria for {event_topic}.",
        expected_output="All the details of a specifically chosen"
                        "venue you found to accommodate the event."
                        "Return valid JSON only, like: {\"website_url\": \"...\"}."
                        "Return a JSON object with numeric 'capacity' (integer only) and 'booking_status' (string: 'available' or 'unavailable').",
        human_input=True,
        output_json=VenueDetails,
        output_file="venue_details.json",  
        # Outputs the venue details as a JSON file
        agent=venue_coordinator
    )

    logistics_task = Task(
        description="Coordinate catering and "
                    "equipment for an event "
                    "with {expected_participants} participants "
                    "on {tentative_date}.",
        expected_output="Confirmation of all logistics arrangements "
                        "including catering and equipment setup.",
        human_input=True,
        async_execution=False,   # The crew must end with at most one asynchronous task
        agent=logistics_manager
    )

    marketing_task = Task(
        description="Promote the {event_topic} "
                    "aiming to engage at least"
                    "{expected_participants} potential attendees.",
        expected_output="Report on marketing activities "
                        "and attendee engagement formatted as markdown.",
        async_execution=True,
        output_file="marketing_report.md",  # Outputs the report as a text file
        agent=marketing_communications_agent
    )
    
    # Define the crew with agents and tasks
    event_management_crew = Crew(
        agents=[venue_coordinator, 
                logistics_manager, 
                marketing_communications_agent],
        
        tasks=[venue_task, 
            logistics_task, 
            marketing_task],
        
        verbose=True
    )

    event_details = {
        'event_topic': "Tech Innovation Conference",
        'event_description': "A gathering of tech innovators "
                            "and industry leaders "
                            "to explore future technologies.",
        'event_city': "San Francisco",
        'tentative_date': "2024-09-15",
        'expected_participants': 500,
        'budget': 20000,
        'venue_type': "Conference Hall"
    }

    result = event_management_crew.kickoff(inputs=event_details)

    with open('venue_details.json') as f:
        data = json.load(f)

    pprint(data)
    
    #Markdown("marketing_report.md")
    with open("marketing_report.md", "r") as f:
        print(f.read())

if __name__ == "__main__":
    main()
