
def format_context(documents: list) -> str:
    context_blocks = []

    for index, doc in enumerate(documents, start=1):
        meta = doc.metadata

        paper_title = meta.get("paper_title", "Unknown paper")
        page_number = meta.get("page_number", "unknown")
        chunk_id = meta.get("chunk_id", "unknown")

        block = (
            f"[Source {index}] {paper_title}, page {page_number}, chunk {chunk_id}\n"
            f"{doc.page_content}"
        )

        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_rag_prompt(question: str, context: str) -> str:
    return f"""
You are a research paper assistant.

Answer the question using only the provided context.
If the answer is not present in the context, say that the context does not contain enough information.
Cite the source number when possible.

Question:
{question}

Context:
{context}

Answer:
""".strip()