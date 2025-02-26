from fastapi import FastAPI, File, UploadFile, Form, Depends
from pydantic import BaseModel
import os
from retriever import create_or_update_faiss, get_retrieved_context
from agents import select_best_agent

app = FastAPI()

UPLOAD_FOLDER = "uploaded_files"
FAISS_INDEX_PATH = "faiss_index"

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(query: str = Form(...), file: UploadFile = None):
    """
    Handles two cases:
    - If only `query` is sent, it uses the multi-agent system.
    - If a `file` + `query` is sent, it updates FAISS and retrieves answers from it.
    """
    # Case 1: If a file is provided, use FAISS
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save file locally
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Read text content from the file
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()

        # Process and store in FAISS
        create_or_update_faiss(text_content)

        # Retrieve answer using FAISS
        retrieved_docs = get_retrieved_context(query)

        if retrieved_docs and retrieved_docs[0][1] > 0:
            response_text = "\n".join([f"Similarity: {score:.2f} - {doc}" for doc, score in retrieved_docs])
            return {"response": response_text}

        return {"response": "File uploaded and processed, but no relevant information found."}

    # Case 2: If no file, use the multi-agent system
    return {"response": select_best_agent(query)}

# Start API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
