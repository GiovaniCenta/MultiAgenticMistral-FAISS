import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# FAISS index directory
FAISS_INDEX_PATH = "faiss_index"

def create_or_update_faiss(text):
    """
    Creates or updates the FAISS index with new text data.
    :param text: The text content to be added to FAISS.
    """

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

    if os.path.exists(FAISS_INDEX_PATH):
        # Load existing FAISS index and add new documents
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        vector_store.add_documents(documents)
    else:
        # Create a new FAISS index
        vector_store = FAISS.from_documents(documents, embedding_model)

    # Save FAISS index
    vector_store.save_local(FAISS_INDEX_PATH)
    print("✅ FAISS index updated successfully!")

def get_retrieved_context(query, k=3, score_threshold=0.5):
    """
    Retrieves relevant documents from FAISS along with similarity scores.
    :param query: User query.
    :param k: Number of results.
    :param score_threshold: Minimum similarity score to return a document.
    :return: List of (document text, similarity score).
    """

    if not os.path.exists(FAISS_INDEX_PATH):
        return [("No knowledge base found. Please upload a file first.", 0)]

    # Load FAISS index
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

    retrieved_docs_with_scores = vector_store.similarity_search_with_score(query, k=k)

    # Filter results based on score threshold
    filtered_results = [
        (doc.page_content, score) 
        for doc, score in retrieved_docs_with_scores 
        if score >= score_threshold
    ]

    return filtered_results
