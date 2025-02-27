from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
import traceback
import argparse
from typing import Optional, Union, Dict, Any
from retriever import create_or_update_faiss, get_retrieved_context
from agents_slm import select_best_agent

app = FastAPI(title="Multi-Agent System")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

UPLOAD_FOLDER = "uploaded_files"
FAISS_INDEX_PATH = "faiss_index"

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class ChatResponse(BaseModel):
    response: Union[str, Dict[str, Any]]
    error: Optional[str] = None

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all unhandled exceptions to prevent server crashes"""
    error_msg = f"Unhandled server error: {str(exc)}"
    return JSONResponse(
        status_code=500,
        content={"response": "Internal Server Error", "error": error_msg}
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(query: str = Form(...), file: Optional[UploadFile] = None):
    """
    Handles two cases:
    - If only `query` is sent, it uses the multi-agent system.
    - If a `file` + `query` is sent, it updates FAISS and retrieves answers from it.
    """
    # Case 1: If a file is provided, use FAISS
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)

            # Save file locally
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Read text content from the file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            except UnicodeDecodeError:
                # Try with a different encoding if UTF-8 fails
                with open(file_path, "r", encoding="latin-1") as f:
                    text_content = f.read()

            # Process and store in FAISS
            create_or_update_faiss(text_content)

            # Use the multi-agent with FAISS enabled
            return ChatResponse(response=select_best_agent(query, use_faiss=True)["output"])
        
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"File processing error: {str(e)}\n{error_trace}")
            return ChatResponse(
                response="Error processing file",
                error=f"File processing error: {str(e)}"
            )

    # Case 2: If no file, use the multi-agent system without FAISS
    try:
        result = select_best_agent(query, use_faiss=False)
        
        # Handle different response types from agents
        if isinstance(result, dict):
            if "output" in result:
                return ChatResponse(response=result["output"])
            else:
                return ChatResponse(response=result)
        else:
            return ChatResponse(response=str(result))
            
    except json.decoder.JSONDecodeError as e:
        error_msg = "Error: The Ollama server returned an invalid response."
        print(f"{error_msg}: {str(e)}")
        return ChatResponse(
            response=error_msg,
            error="Make sure Ollama is running with 'ollama serve' and the phi model is downloaded with 'ollama pull phi'"
        )
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Agent error: {str(e)}\n{error_trace}")
        return ChatResponse(
            response="Error processing query with multi-agent system",
            error=f"Agent error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    try:
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def find_available_port(start_port, max_attempts=10):
    """Try to find an available port, starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to create a socket binding
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    
    # If no ports are available in the range
    return None

# Start API
if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start the Multi-Agent RAG API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--auto-port", action="store_true", help="Automatically find an available port if the specified one is in use")
    
    args = parser.parse_args()
    
    # Setup for auto port selection if requested
    port = args.port
    if args.auto_port:
        available_port = find_available_port(args.port)
        if available_port:
            port = available_port
            print(f"Port {args.port} is in use. Using port {port} instead.")
        else:
            print(f"Could not find an available port in range {args.port}-{args.port+10}. Please specify a different port.")
            exit(1)
    
    print(f"Starting Multi-Agent RAG API server on {args.host}:{port}...")
    print("Make sure Ollama is running with: ollama serve")
    print("Make sure you have downloaded the phi model with: ollama pull phi")
    
    try:
        uvicorn.run(app, host=args.host, port=port)
    except OSError as e:
        if "address already in use" in str(e).lower() or "endereço de soquete" in str(e).lower():
            print(f"\nERROR: Port {port} is already in use.")
            print(f"Try running with a different port: python main.py --port 8001")
            print(f"Or use automatic port selection: python main.py --auto-port")
        else:
            print(f"Error starting server: {e}")
    except Exception as e:
        print(f"Error starting server: {e}")