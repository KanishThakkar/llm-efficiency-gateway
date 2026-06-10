from langchain_text_splitters import RecursiveCharacterTextSplitter

from efficiency_gateway.core.token_counter import TokenCounter


def chunk_documents(
    documents,
    chunk_size: int = 1800,
    chunk_overlap: int = 250,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    token_counter = TokenCounter()

    for index, chunk in enumerate(chunks):
        paper_id = chunk.metadata.get("paper_id", "unknown_paper")

        chunk.metadata["chunk_id"] = f"{paper_id}_chunk_{index}"
        chunk.metadata["chunk_index"] = index
        chunk.metadata["token_count"] = token_counter.count(chunk.page_content).tokens

    return chunks