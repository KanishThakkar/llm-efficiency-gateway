from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


PAPER_TITLES = {
    "attention_is_all_you_need.pdf": "Attention Is All You Need",
    "rag_for_knowledge_intensive_nlp.pdf": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "prompt_compression.pdf": "Prompt Compression",
    "quantization.pdf": "Quantization",
}


def paper_id_from_path(pdf_path: Path) -> str:
    return pdf_path.stem.lower().replace(" ", "_").replace("-", "_")


def load_pdf_documents(pdf_path: str | Path):
    pdf_path = Path(pdf_path)

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    paper_id = paper_id_from_path(pdf_path)
    paper_title = PAPER_TITLES.get(pdf_path.name, pdf_path.stem.replace("_", " ").title())

    for doc in docs:
        doc.metadata["paper_id"] = paper_id
        doc.metadata["paper_title"] = paper_title
        doc.metadata["source_file"] = pdf_path.name

        if "page" in doc.metadata:
            doc.metadata["page_number"] = int(doc.metadata["page"]) + 1

    return docs


def load_papers_from_folder(folder_path: str | Path):
    folder_path = Path(folder_path)

    all_docs = []

    for pdf_path in sorted(folder_path.glob("*.pdf")):
        docs = load_pdf_documents(pdf_path)
        all_docs.extend(docs)

    return all_docs