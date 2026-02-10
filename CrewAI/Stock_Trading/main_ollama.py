import warnings
import os
# Must be before ANY crewai import
os.environ["CREWAI_TELEMETRY"] = "0"
os.environ["CREWAI_TRACING"] = "0"
os.environ["CREWAI_SENTRY"] = "0"
os.environ["CREWAI_PARALLEL_THINKING"] = "0"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = ""
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = ""
from crewai import Agent, Task, Crew, LLM, Process
from utils.get_serper_api_key import get_serper_api_key
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
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

    # Critical: Remove OpenAI env vars so CrewAI won't force OpenAI mode
    for var in [
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_API_HOST",
        "OPENAI_BASE_URL",
        "OPENAI_KEY",
    ]:
        os.environ.pop(var, None)

    # Use Ollama (via OpenAI-compatible proxy)
    os.environ["OPENAI_API_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"  # dummy key since Ollama runs locally
    os.environ["SERPER_API_KEY"] = get_serper_api_key()

    llm = LLM(
        model="llama3.2:3b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
#        allow_empty_output=True,
        timeout=120
    )

    print("LLM Base URL:", llm.client.base_url)

    # llm = Ollama(
    #     model="llama3.2:3b",
    #     base_url="http://localhost:11434/v1"
    # )

    # Initialize the tools
    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()

    # Agent 1: Data Analyst
    data_analyst_agent = Agent(
        role="Data Analyst",
        goal=(
            "You MUST provide precise, factual, structured financial analysis. "
            "ONLY return clean, concise, well-reasoned content. "
            "You MUST NEVER imitate websites, UI layouts, tables of contents, "
            "ads, or Wikipedia/articles/blogs. "
            "You MUST NOT output HTML, markdown templates, headings like "
            "'Edit Article'/'Read History', or anything resembling a webpage."
        ),
        backstory=(
            "You are a highly specialized financial intelligence system. "
            "You think step-by-step, verify facts, and present only useful "
            "financial insights. You strictly avoid any irrelevant text or "
            "fictional formatting."
        ),
        verbose=True,
        allow_delegation=True,
        tools=[scrape_tool, search_tool],
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
            "You MUST produce a concise, factual market analysis. "
            "Rules:\n"
            "1. Do NOT output anything resembling a webpage.\n"
            "2. Do NOT invent sections or headings.\n"
            "3. Keep it analytical and structured.\n\n"
            "TASK:\n"
            "Analyze real-time and historical market data for {stock_selection}. "
            "Identify patterns, important levels, volatility changes, sentiment, "
            "momentum shifts, and any signals relevant for trading."
        ),
        expected_output=(
            "A structured analysis including:\n"
            "- Trend direction\n"
            "- Key price levels\n"
            "- Volatility summary\n"
            "- Short-term forecast signals\n"
            "- Any notable opportunities or threats"
        ),
        agent=data_analyst_agent,
    )


    # Task for Trading Strategy Agent: Develop Trading Strategies
    strategy_development_task = Task(
        description=(
            "You MUST output a clean, structured set of trading strategies.\n"
            "Rules:\n"
            "1. NO webpage-like text.\n"
            "2. NO irrelevant sections.\n"
            "3. NO HTML/markdown templates.\n\n"
            "TASK:\n"
            "Based on insights from the analyst and the user's parameters "
            "(risk tolerance: {risk_tolerance}, strategy preference: {trading_strategy_preference}), "
            "develop several actionable trading strategies for {stock_selection}."
        ),
        expected_output=(
            "A concise list of strategy options including:\n"
            "- Strategy name\n"
            "- Description\n"
            "- Entry logic\n"
            "- Exit logic\n"
            "- Risk management rules"
        ),
        agent=trading_strategy_agent,
    )


    # Task for Trade Advisor Agent: Plan Trade Execution
    execution_planning_task = Task(
        description=(
            "Produce a focused trade execution plan.\n"
            "Rules:\n"
            "1. NO webpage-like headings.\n"
            "2. NO decorative content.\n"
            "3. Keep output actionable and relevant.\n\n"
            "TASK:\n"
            "Given the approved strategies for {stock_selection}, outline the "
            "optimal execution timing, order type selection, liquidity "
            "considerations, and expected slippage impact."
        ),
        expected_output=(
            "A short execution plan including:\n"
            "- Best entry timing\n"
            "- Best exit timing\n"
            "- Recommended order types\n"
            "- Liquidity considerations\n"
            "- Slippage expectations"
        ),
        agent=execution_agent,
    )


    # Task for Risk Advisor Agent: Assess Trading Risks
    risk_assessment_task = Task(
        description=(
            "Provide a clear, direct risk analysis.\n"
            "Rules:\n"
            "1. NO webpage-like text.\n"
            "2. NO filler or disclaimers.\n"
            "3. Be precise and analytical.\n\n"
            "TASK:\n"
            "Evaluate the risks of the proposed strategies and execution plan "
            "for {stock_selection}. Include risk levels, mitigation steps, and "
            "capital exposure considerations."
        ),
        expected_output=(
            "A structured risk report including:\n"
            "- Main risks\n"
            "- Severity\n"
            "- Probability\n"
            "- Mitigation actions\n"
            "- Recommended constraints"
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
        
        manager_llm = LLM(
            model="llama3.2:3b",
            temperature=0.7,
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        ),
#        process=Process.hierarchical,    # Hierarchical requires strong models (like 70B or GPT-4 level)
        process=Process.sequential,
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
    Markdown(result.raw)


if __name__ == "__main__":
    main()
