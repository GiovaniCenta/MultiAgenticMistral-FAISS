from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper
import re
from retriever import get_retrieved_context  # FAISS retrieval function

# Load Phi-2 with Ollama
llm = ChatOllama(model="phi")

# Wikipedia Search Agent
wiki = WikipediaAPIWrapper()
def search_wikipedia(query: str):       
    """Searches Wikipedia for relevant information."""
    return wiki.run(query)

# Summarization Agent
def summarize_text(text: str):
    """Summarizes long texts into a short, concise format."""
    return llm.predict(f"Think step by step before answering: Summarize this: {text}")

# Code Generation Agent
def code_helper(prompt: str):
    """Generates Python code based on a prompt."""
    return llm.predict(f"Write a Python script for: {prompt}")

# Translation Agent
def translate_text(text: str, language: str = "French"):
    """Translates text into the specified language (default: French)."""
    return llm.predict(f"Translate this text to {language}: {text}")

# Sentiment Analysis Agent
def sentiment_analysis(text: str):
    """Analyzes the sentiment of a given text."""
    return llm.predict(f"Analyze the sentiment of this text: {text}")

# Code Explanation Agent
def explain_code(code: str):
    """Explains a Python code snippet in simple terms."""
    return llm.predict(f"Explain this Python code: {code}")

# Text Rewriter Agent
def rewrite_text(text: str, style: str = "formal"):
    """Rewrites a given text in a different style (default: formal)."""
    return llm.predict(f"Rewrite this text in a {style} style: {text}")

# Python Docs Agent
def python_docs(query: str):
    """Answers questions about Python documentation."""
    return llm.predict(f"Answer this based on Python documentation: {query}")

# FAISS + RAG Integration
def knowledge_retrieval(query: str):
    """Retrieves relevant information from FAISS vector database."""
    return get_retrieved_context(query)

# Structured Chat Agent (for APIs and structured outputs)
def structured_response(query: str):
    """Returns structured responses (useful for APIs, JSON outputs, etc.)."""
    return llm.predict(f"Format the following response as structured JSON: {query}")

# Self-Ask Agent (Breaks down complex queries into smaller ones)
def self_ask(query: str):
    """Breaks down the query into smaller sub-questions before answering."""
    return llm.predict(f"Break this question into smaller steps before answering: {query}")

# Registering Agents as LangChain Tools
tools = [
    Tool(name="Search", func=search_wikipedia, description="Search Wikipedia for relevant information."),
    Tool(name="Summarizer", func=summarize_text, description="Summarizes long texts."),
    Tool(name="Code Generator", func=code_helper, description="Generates Python code."),
    Tool(name="Translator", func=translate_text, description="Translates text into different languages."),
    Tool(name="Sentiment Analysis", func=sentiment_analysis, description="Analyzes the sentiment of a text."),
    Tool(name="Code Explainer", func=explain_code, description="Explains Python code."),
    Tool(name="Text Rewriter", func=rewrite_text, description="Rewrites text in different styles."),
    Tool(name="Python Docs", func=python_docs, description="Answers questions about Python documentation."),
    Tool(name="Knowledge Retrieval", func=knowledge_retrieval, description="Retrieves information from FAISS database."),
    Tool(name="Structured Chat", func=structured_response, description="Formats the response into structured JSON.")
]

# Self-Ask Agent requires exactly one tool
search_tool = Tool(name="Intermediate Answer", func=search_wikipedia, description="Search Wikipedia for relevant information.")

# Memory for Conversational Context
memory = ConversationBufferMemory()

# Multi-Agent System
# Default agent: ZERO_SHOT_REACT_DESCRIPTION
default_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# New Agent: STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION (For structured outputs, like JSON)
structured_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# New Agent: SELF_ASK_WITH_SEARCH (For breaking complex questions)
self_ask_agent = initialize_agent(
    tools=[search_tool],  # Only one tool allowed
    llm=llm,
    agent=AgentType.SELF_ASK_WITH_SEARCH,
    verbose=True
)

# Intent Detection (Direct Routing)
def detect_intent(query):
    if re.search(r'\btranslate\b|\bto (Spanish|French|German)\b', query, re.IGNORECASE):
        return "Translator"
    elif re.search(r'\bsentiment\b|\banalyze\b', query, re.IGNORECASE):
        return "Sentiment Analysis"
    elif re.search(r'\bexplain\b|\bcode\b', query, re.IGNORECASE):
        return "Code Explainer"
    elif re.search(r'\brewrite\b|\bparaphrase\b', query, re.IGNORECASE):
        return "Text Rewriter"
    return None

# Agent Selection
def select_best_agent(query):
    """Chooses the best agent based on query type."""
    
    # Detect requests for structured output
    if re.search(r'\bjson\b|\btable\b|\bstructured\b', query, re.IGNORECASE):
        return structured_agent.invoke({"input": query})

    # Detect complex/multi-step questions
    if re.search(r'\bhow\b|\bwhy\b|\bexplain\b|\bbreak down\b', query, re.IGNORECASE):
        return self_ask_agent.invoke({"input": query})

    # Directly route based on intent detection
    intent = detect_intent(query)
    if intent:
        return default_agent.invoke({"input": f"Use the {intent} tool: {query}"})

    # Default to general-purpose agent
    return default_agent.invoke({"input": query})
