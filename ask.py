import requests
import argparse

API_URL = "http://localhost:8000/chat"

def ask_question(query, file_path=None):
    """Sends a query to the chatbot API, optionally with a file for FAISS retrieval."""
    data = {"query": query}

    if file_path:
        with open(file_path, "rb") as file:
            response = requests.post(API_URL, files={"file": file}, data=data)
    else:
        response = requests.post(API_URL, data=data)

    return response.json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions to the chatbot.")
    parser.add_argument("query", type=str, help="The question to ask.")
    parser.add_argument("--file", type=str, help="Optional file for FAISS retrieval.")

    args = parser.parse_args()

    result = ask_question(args.query, args.file)
    print(result["response"])
