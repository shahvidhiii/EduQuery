import uvicorn

from src.eduquery.app import API_HOST, API_PORT, EMBEDDING_MODEL, LLM_MODEL, USE_CHROMADB, app


if __name__ == "__main__":
    print(f"Backend: {'ChromaDB' if USE_CHROMADB else 'Pickle'}")
    print(f"Embedding Model: {EMBEDDING_MODEL} | LLM: {LLM_MODEL}")
    print(f"Visit http://{API_HOST}:{API_PORT}/docs for the API playground.")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
