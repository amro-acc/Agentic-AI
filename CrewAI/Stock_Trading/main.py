import warnings
import os
from crewai import Agent, Task, Crew, LLM, Process
from utils.get_groq_api_key import get_groq_api_key
from utils.get_serper_api_key import get_serper_api_key
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain_openai import ChatOpenAI
from IPython.display import Markdown


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

    # Agent 1: Data Analyst
    data_analyst_agent = Agent(
        role="Data Analyst",
        goal="Monitor and analyze market data in real-time "
            "to identify trends and predict market movements.",
        backstory="Specializing in financial markets, this agent "
                "uses statistical modeling and machine learning "
                "to provide crucial insights. With a knack for data, "
                "the Data Analyst Agent is the cornerstone for "
                "informing trading decisions.",
        verbose=True,
        allow_delegation=True,
        tools = [scrape_tool, search_tool],
        llm=llm
    )

    # Agent 2: Trading Strategy Developer
    trading_strategy_agent = Agent(
        role="Trading Strategy Developer",
        goal="Develop and test various trading strategies based "
            "on insights from the Data Analyst Agent.",
        backstory="Equipped with a deep understanding of financial "
                "markets and quantitative analysis, this agent "
                "devises and refines trading strategies. It evaluates "
                "the performance of different approaches to determine "
                "the most profitable and risk-averse options.",
        verbose=True,
        allow_delegation=True,
        tools = [scrape_tool, search_tool],
        llm=llm
    )

    # Agent 3: Trade Advisor
    execution_agent = Agent(
        role="Trade Advisor",
        goal="Suggest optimal trade execution strategies "
            "based on approved trading strategies.",
        backstory="This agent specializes in analyzing the timing, price, "
                "and logistical details of potential trades. By evaluating "
                "these factors, it provides well-founded suggestions for "
                "when and how trades should be executed to maximize "
                "efficiency and adherence to strategy.",
        verbose=True,
        allow_delegation=True,
        tools = [scrape_tool, search_tool],
        llm=llm
    )

    # Agent 4: Risk Advisor
    risk_management_agent = Agent(
        role="Risk Advisor",
        goal="Evaluate and provide insights on the risks "
            "associated with potential trading activities.",
        backstory="Armed with a deep understanding of risk assessment models "
                "and market dynamics, this agent scrutinizes the potential "
                "risks of proposed trades. It offers a detailed analysis of "
                "risk exposure and suggests safeguards to ensure that "
                "trading activities align with the firm’s risk tolerance.",
        verbose=True,
        allow_delegation=True,
        tools = [scrape_tool, search_tool],
        llm=llm
    )

    # Task for Data Analyst Agent: Analyze Market Data
    data_analysis_task = Task(
        description=(
            "Continuously monitor and analyze market data for "
            "the selected stock ({stock_selection}). "
            "Use statistical modeling and machine learning to "
            "identify trends and predict market movements."
        ),
        expected_output=(
            "Insights and alerts about significant market "
            "opportunities or threats for {stock_selection}."
        ),
        agent=data_analyst_agent,
    )

    # Task for Trading Strategy Agent: Develop Trading Strategies
    strategy_development_task = Task(
        description=(
            "Develop and refine trading strategies based on "
            "the insights from the Data Analyst and "
            "user-defined risk tolerance ({risk_tolerance}). "
            "Consider trading preferences ({trading_strategy_preference})."
        ),
        expected_output=(
            "A set of potential trading strategies for {stock_selection} "
            "that align with the user's risk tolerance."
        ),
        agent=trading_strategy_agent,
    )

    # Task for Trade Advisor Agent: Plan Trade Execution
    execution_planning_task = Task(
        description=(
            "Analyze approved trading strategies to determine the "
            "best execution methods for {stock_selection}, "
            "considering current market conditions and optimal pricing."
        ),
        expected_output=(
            "Detailed execution plans suggesting how and when to "
            "execute trades for {stock_selection}."
        ),
        agent=execution_agent,
    )

    # Task for Risk Advisor Agent: Assess Trading Risks
    risk_assessment_task = Task(
        description=(
            "Evaluate the risks associated with the proposed trading "
            "strategies and execution plans for {stock_selection}. "
            "Provide a detailed analysis of potential risks "
            "and suggest mitigation strategies."
        ),
        expected_output=(
            "A comprehensive risk analysis report detailing potential "
            "risks and mitigation recommendations for {stock_selection}."
        ),
        agent=risk_management_agent,
    )

    # Define the crew with agents and tasks
    financial_trading_crew = Crew(
        agents=[data_analyst_agent, 
                trading_strategy_agent, 
                execution_agent, 
                risk_management_agent],
        
        tasks=[data_analysis_task, 
            strategy_development_task, 
            execution_planning_task, 
            risk_assessment_task],
        
        manager_llm=ChatOpenAI(model="groq/llama-3.3-70b-versatile", 
                            temperature=0.7),
        process=Process.hierarchical,
        verbose=True
    )

    # Example data for kicking off the process
    financial_trading_inputs = {
        'stock_selection': 'AAPL',
        'initial_capital': '100000',
        'risk_tolerance': 'Medium',
        'trading_strategy_preference': 'Day Trading',
        'news_impact_consideration': True
    }

    ### this execution will take some time to run
    result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)

    # Display the final result as Markdown
    Markdown(result)


if __name__ == "__main__":
    main()
