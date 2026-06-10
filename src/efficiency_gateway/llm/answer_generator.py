import time
from dataclasses import dataclass

from litellm import completion

from efficiency_gateway.core.token_counter import TokenCounter

# Maps internal model names -> LiteLLM model strings
_MODEL_MAP: dict[str, str] = {
    "local-quantized": "ollama/mistral",
}


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model_used: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMAnswerGenerator:
    def __init__(
        self,
        ollama_api_base: str = "http://localhost:11434",
        timeout: int = 60,
        max_tokens: int = 800,
    ):
        self.ollama_api_base = ollama_api_base
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.token_counter = TokenCounter()

    def generate(self, model: str, prompt: str) -> GenerationResult:
        litellm_model = _MODEL_MAP.get(model, model)

        kwargs: dict = {
            "model": litellm_model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
        }

        if litellm_model.startswith("ollama/"):
            kwargs["api_base"] = self.ollama_api_base

        start = time.perf_counter()
        response = completion(**kwargs)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        answer: str = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)

        input_tokens = int(usage.prompt_tokens) if usage else self.token_counter.count(prompt).tokens
        output_tokens = int(usage.completion_tokens) if usage else self.token_counter.count(answer).tokens

        return GenerationResult(
            answer=answer,
            model_used=litellm_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )
