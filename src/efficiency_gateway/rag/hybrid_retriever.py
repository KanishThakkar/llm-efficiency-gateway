from pathlib import Path

import chromadb
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class HybridPaperRetriever:
    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "research_papers",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        bm25_k: int = 8,
        vector_k: int = 8,
        bm25_weight: float = 0.45,
        vector_weight: float = 0.55,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
        )

        self.vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=str(self.persist_dir),
            embedding_function=self.embedding,
        )

        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": vector_k}
        )

        bm25_documents = self._load_documents_from_chroma()

        if not bm25_documents:
            raise ValueError(
                "No documents found in ChromaDB. Run scripts/ingest_papers_step2.py first."
            )

        self.bm25_retriever = BM25Retriever.from_documents(bm25_documents)
        self.bm25_retriever.k = bm25_k

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[bm25_weight, vector_weight],
        )

    def _load_documents_from_chroma(self) -> list[Document]:
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection = client.get_collection(self.collection_name)
        records = collection.get(include=["documents", "metadatas"])

        texts = records.get("documents", [])
        metadatas = records.get("metadatas", [])

        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(texts, metadatas)
            if text
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        docs = self.hybrid_retriever.invoke(query)
        return docs[:top_k]
