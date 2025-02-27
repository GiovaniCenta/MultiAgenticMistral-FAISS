from langchain_ollama import ChatOllama
from langchain_community.utilities import WikipediaAPIWrapper
import re
import json
import requests
import traceback
from retriever import get_retrieved_context  # FAISS retrieval function

# Load Phi-2 with Ollama - Add error handling and timeout
try:
    # Add base_url if Ollama is not running on default localhost:11434
    llm = ChatOllama(
        model="phi",
        temperature=0.7,
        timeout=30  # Add timeout to prevent hanging
    )
except Exception as e:
    print(f"Error initializing Ollama: {e}")
    print("Make sure Ollama is running with 'ollama serve' and the phi model is downloaded")
    exit(1)

# Wikipedia Search Agent with error handling
wiki = WikipediaAPIWrapper()
def search_wikipedia(query: str):
    """Searches Wikipedia for relevant information."""
    try:
        return wiki.run(query)
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

# Summarization Agent with error handling
def summarize_text(text: str):
    """Summarizes long texts into a short, concise format."""
    try:
        return llm.invoke(f"Think step by step before answering: Summarize this: {text}").content
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

# Code Generation Agent with error handling
def code_helper(prompt: str):
    """Generates Python code based on a prompt."""
    try:
        return llm.invoke(f"Write a Python script for: {prompt}").content
    except Exception as e:
        return f"Error generating code: {str(e)}"

# Translation Agent with error handling
def translate_text(text: str, language: str = "French"):
    """Translates text into the specified language (default: French)."""
    try:
        # Extract language from the input if provided
        match = re.search(r'to\s+(\w+)', text, re.IGNORECASE)
        if match:
            language = match.group(1)
            text = text.replace(match.group(0), "").strip()
        
        return llm.invoke(f"Translate this text to {language}: {text}").content
    except requests.exceptions.JSONDecodeError as e:
        return f"Error: Ollama returned an invalid response. Ensure Ollama is running properly with the phi model loaded."
    except Exception as e:
        return f"Error translating text: {str(e)}"

# Sentiment Analysis Agent with error handling
def sentiment_analysis(text: str):
    """Analyzes the sentiment of a given text."""
    try:
        return llm.invoke(f"Analyze the sentiment of this text: {text}").content
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

# Code Explanation Agent with error handling
def explain_code(code: str):
    """Explains a Python code snippet in simple terms."""
    try:
        return llm.invoke(f"Explain this Python code: {code}").content
    except Exception as e:
        return f"Error explaining code: {str(e)}"

# Text Rewriter Agent with error handling
def rewrite_text(text: str, style: str = "formal"):
    """Rewrites a given text in a different style (default: formal)."""
    try:
        # Extract style from the input if provided
        match = re.search(r'in\s+(\w+)\s+style', text, re.IGNORECASE)
        if match:
            style = match.group(1)
            text = text.replace(match.group(0), "").strip()
            
        return llm.invoke(f"Rewrite this text in a {style} style: {text}").content
    except Exception as e:
        return f"Error rewriting text: {str(e)}"

# Python Docs Agent with error handling
def python_docs(query: str):
    """Answers questions about Python documentation."""
    try:
        return llm.invoke(f"Answer this based on Python documentation: {query}").content
    except Exception as e:
        return f"Error querying Python docs: {str(e)}"

# FAISS + RAG Integration with error handling
def knowledge_retrieval(query: str):
    """Retrieves relevant information from FAISS vector database."""
    try:
        return get_retrieved_context(query)
    except Exception as e:
        return f"Error retrieving knowledge: {str(e)}"

# Structured Chat Agent with error handling
def structured_response(query: str):
    """Returns structured responses (useful for APIs, JSON outputs, etc.)."""
    try:
        return llm.invoke(f"Format the following response as structured JSON: {query}").content
    except Exception as e:
        return f"Error creating structured response: {str(e)}"

# Self-Ask Agent with error handling
def self_ask(query: str):
    """Breaks down the query into smaller sub-questions before answering."""
    try:
        return llm.invoke(f"Break this question into smaller steps before answering: {query}").content
    except Exception as e:
        return f"Error processing complex query: {str(e)}"

# Direct query execution - our main approach now
def direct_query(query: str):
    """Direct execution without agent framework."""
    try:
        return llm.invoke(query).content
    except Exception as e:
        return f"Error with direct query: {str(e)}"

# Smart Router - a simplified approach that doesn't use LangChain's agent framework
def select_best_agent(query, use_faiss=False):
    """
    Routes queries to the appropriate service based on intent detection.
    
    Args:
        query (str): The user's query
        use_faiss (bool): Whether to use FAISS for retrieval (typically True when a file was uploaded)
    """
    try:
        # Initialize response
        response = ""
        
        # 1. Check for translation requests
        if re.search(r'\btranslate\b|\bto (Spanish|French|German|Italian|Portuguese)\b', query, re.IGNORECASE):
            print("Translation request detected")
            response = translate_text(query)
            
        # 2. Check for sentiment analysis
        elif re.search(r'\bsentiment\b|\banalyze sentiment\b|\bhow does.*feel\b', query, re.IGNORECASE):
            print("Sentiment analysis request detected")
            # Extract the text to analyze
            match = re.search(r'sentiment of "(.*?)"', query, re.IGNORECASE)
            if match:
                text_to_analyze = match.group(1)
            else:
                text_to_analyze = re.sub(r'(analyze|sentiment|the sentiment of)', '', query, flags=re.IGNORECASE).strip()
            response = sentiment_analysis(text_to_analyze)
            
        # 3. Check for code explanation
        elif re.search(r'\bexplain\s+code\b|\bexplain\s+this\s+code\b', query, re.IGNORECASE):
            print("Code explanation request detected")
            # Try to extract code from the query using regex
            code_match = re.search(r'```(.*?)```', query, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                code = query.replace("explain code", "").replace("explain this code", "").strip()
            response = explain_code(code)
            
        # 4. Check for code generation
        elif re.search(r'\bwrite\s+code\b|\bgenerate\s+code\b|\bcode\s+for\b', query, re.IGNORECASE):
            print("Code generation request detected")
            prompt = re.sub(r'(write|generate|create)\s+code\s+(for|to)', '', query, flags=re.IGNORECASE).strip()
            response = code_helper(prompt)
            
        # 5. Check for text rewriting
        elif re.search(r'\brewrite\b|\bparaphrase\b|\bin\s+\w+\s+style\b', query, re.IGNORECASE):
            print("Text rewriting request detected")
            response = rewrite_text(query)
            
        # 6. Check for summarization
        elif re.search(r'\bsummarize\b|\bsummary\b', query, re.IGNORECASE):
            print("Summarization request detected")
            text_to_summarize = re.sub(r'(summarize|summary|provide a summary of)', '', query, flags=re.IGNORECASE).strip()
            response = summarize_text(text_to_summarize)
            
        # 7. Check for knowledge retrieval/search
        elif re.search(r'\bsearch\b|\blookup\b|\bfind information\b', query, re.IGNORECASE):
            print("Search request detected")
            search_query = re.sub(r'(search|lookup|find information about|find|information about)', '', query, flags=re.IGNORECASE).strip()
            response = search_wikipedia(search_query)
            
        # 8. Check for JSON/structured output requests
        elif re.search(r'\bjson\b|\bstructured\b|\bformat as json\b', query, re.IGNORECASE):
            print("Structured output request detected")
            response = structured_response(query)
            
        # 9. Default: Direct query to LLM with optional FAISS
        else:
            print("Using direct query to LLM")
            
            # Only use FAISS if explicitly requested (when a file was uploaded)
            if use_faiss:
                try:
                    context = get_retrieved_context(query)
                    if context and len(context) > 0:
                        print("Retrieved context from FAISS")
                        context_text = "\n".join([doc for doc, score in context])
                        # Use the context to answer the query
                        response = llm.invoke(f"Use the following information to answer the question: '{query}'\n\nInformation:\n{context_text}").content
                    else:
                        # No relevant context found, use direct query
                        response = direct_query(query)
                except Exception as e:
                    print(f"FAISS retrieval failed: {e}, falling back to direct query")
                    response = direct_query(query)
            else:
                # Skip FAISS and go directly to LLM
                response = direct_query(query)
        
        # Ensure we return a valid response
        if not response or response.strip() == "":
            response = "I couldn't process your request. Could you please try rephrasing it?"
            
        return {"output": response}
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in select_best_agent: {str(e)}\n{error_trace}")
        return {"output": f"I encountered an error while processing your request: {str(e)}. Please try again with a simpler query."}

# Main entry point for direct testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Processing query: {query}")
        try:
            result = select_best_agent(query)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Please provide a query as command-line argument.")