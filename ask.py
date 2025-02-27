#!/usr/bin/env python
"""
Client script for the multi-agent FastAPI system.
Usage: python ask.py "your query here" [--file your_file.txt] [--url http://localhost:8001/chat]
"""

import requests
import argparse
import sys
import json
import os

# Default API URL - can be overridden with --url parameter
DEFAULT_API_URL = "http://localhost:8000/chat"

def ask_question(query, file_path=None, api_url=DEFAULT_API_URL):
    """Sends a query to the chatbot API, optionally with a file for FAISS retrieval."""
    data = {"query": query}
    
    try:
        if file_path:
            with open(file_path, "rb") as file:
                response = requests.post(api_url, files={"file": file}, data=data)
        else:
            response = requests.post(api_url, data=data)
        
        # Handle HTTP errors
        response.raise_for_status()
        
        # Try to parse JSON response
        try:
            return response.json()
        except json.JSONDecodeError:
            print("ERROR: Server returned invalid JSON")
            print("Raw response:", response.text[:200] + "..." if len(response.text) > 200 else response.text)
            print("\nThis likely means the Ollama server is not running or responding correctly")
            return {"response": "Error: Invalid response from server"}
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the server at", api_url)
        print("Make sure the FastAPI server is running (python main.py)")
        print("If the server is running on a different port, use the --url parameter")
        return {"response": "Error: Could not connect to server"}
        
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error occurred: {e}")
        return {"response": f"Error: HTTP {response.status_code}"}
        
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        return {"response": f"Error: {str(e)}"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions to the chatbot.")
    parser.add_argument("query", type=str, nargs="+", help="The question to ask.")
    parser.add_argument("--file", type=str, help="Optional file for FAISS retrieval.")
    parser.add_argument("--url", type=str, default=DEFAULT_API_URL, 
                       help=f"API URL (default: {DEFAULT_API_URL})")
    
    # Get URL from environment variable if present
    if "CHATBOT_API_URL" in os.environ:
        default_url = os.environ["CHATBOT_API_URL"]
        parser.set_defaults(url=default_url)

    args = parser.parse_args()
    
    # Join multiple words into a single query
    query = " ".join(args.query)
    
    print(f"Sending query to {args.url}...")
    result = ask_question(query, args.file, args.url)
    
    # Pretty print the response
    if isinstance(result["response"], dict):
        print(json.dumps(result["response"], indent=2))
    else:
        print(result["response"])
    
    # Print error if present
    if "error" in result and result["error"]:
        print("\nERROR:", result["error"])