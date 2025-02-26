### Simple project of a MultiAgent RAG using Phi2 


-- The purpose is to be a summary of the main tools to do stuff with LLMs and SLMs

# Main tools:
### Ollama
Ollama: is a tool that allows running LLMs locally, optimizing performance for limited hardware. It provides an easy way to download, run, and integrate models like Phi-2 without relying on cloud services. Ideal for privacy-focused and cost-free AI applications.

### Langchain
LangChain: is a framework that simplifies working with LLMs by providing tools for memory, retrieval, and multi-agent orchestration. It enables seamless integration with APIs, vector databases, and local models like Phi-2. Ideal for building advanced AI applications with minimal effort.
 
### FAISS  
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It enables fast retrieval of relevant documents by storing and searching embeddings. Ideal for implementing Retrieval-Augmented Generation (RAG) in AI applications.

python main.py 
python ask.py "Whats is life?"    ## To use the agents
python ask.py "What is kubernets?" --file "kubernets.txt"    ## To use RAG with FAISS

