from dataclasses import dataclass

from efficiency_gateway.core.token_counter import TokenCounter


@dataclass(frozen=True)
class CompressionResult:
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    compression_ratio: float
    compressor_name: str


class LLMLinguaPromptCompressor:
    """
    Compresses prompts with LLMLingua-2 when available.
    Falls back to a no-op compressor if LLMLingua is not installed or fails to load,
    so the rest of the pipeline always works.
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        rate: float = 0.55,
    ):
        self.model_name = model_name
        self.rate = rate
        self.token_counter = TokenCounter()
        self._compressor = self._try_load_llmlingua(model_name)

    def _try_load_llmlingua(self, model_name: str):
        try:
            from llmlingua import PromptCompressor
            comp = PromptCompressor(model_name=model_name, use_llmlingua2=True)
            print(f"[compressor] LLMLingua-2 loaded: {model_name}")
            return comp
        except Exception as exc:
            print(f"[compressor] LLMLingua not available ({exc}). Using no-op fallback.")
            return None

    def compress_text(self, text: str) -> CompressionResult:
        original_tokens = self.token_counter.count(text).tokens

        if self._compressor is None:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                tokens_saved=0,
                compression_ratio=1.0,
                compressor_name="no-op-fallback",
            )

        try:
            result = self._compressor.compress_prompt(
                text,
                rate=self.rate,
                force_tokens=["\n", ".", "?", ":"],
            )
            compressed_text = result.get("compressed_prompt", text)
        except Exception as exc:
            print(f"[compressor] Compression failed ({exc}). Returning original text.")
            compressed_text = text

        compressed_tokens = self.token_counter.count(compressed_text).tokens
        tokens_saved = original_tokens - compressed_tokens
        compression_ratio = (
            round(original_tokens / compressed_tokens, 4) if compressed_tokens else 1.0
        )

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            compression_ratio=compression_ratio,
            compressor_name=f"LLMLingua2:{self.model_name}",
        )
