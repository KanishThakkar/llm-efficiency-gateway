import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficiency_gateway.rag.chunker import chunk_documents
from efficiency_gateway.rag.document_loader import load_papers_from_folder
from efficiency_gateway.rag.vector_store import PaperVectorStore


PAPERS_DIR = ROOT / "data" / "papers"


def main():
    print("Loading PDFs...")
    documents = load_papers_from_folder(PAPERS_DIR)
    print(f"Loaded pages: {len(documents)}")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created chunks: {len(chunks)}")

    print("Saving chunks to ChromaDB...")
    vector_store = PaperVectorStore(
        persist_dir=str(ROOT / "data" / "chroma"),
        collection_name="research_papers",
    )

    vector_store.upsert_documents(chunks)

    print("Ingestion complete.")
    print(f"ChromaDB chunk count: {vector_store.count()}")


if __name__ == "__main__":
    main()