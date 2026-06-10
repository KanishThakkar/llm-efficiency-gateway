import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class TokenCount:
    text: str
    tokens: int
    tokenizer: str


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self.encoding = self._load_tiktoken_encoding()

    def count(self, text: str) -> TokenCount:
        if self.encoding:
            tokens = len(self.encoding.encode(text))
            return TokenCount(text=text, tokens=tokens, tokenizer=f"tiktoken:{self.encoding_name}")

        tokens = len(TOKEN_PATTERN.findall(text))
        return TokenCount(text=text, tokens=tokens, tokenizer="regex-fallback")

    def _load_tiktoken_encoding(self):
        try:
            import tiktoken
            return tiktoken.get_encoding(self.encoding_name)
        except Exception:
            return None