"""
Lightweight quality evaluator using only Python built-ins (difflib + collections).
No model loading — works on any machine regardless of RAM.

Metrics:
  faithfulness      — how much of the answer text overlaps with the retrieved context
  answer_relevancy  — how much of the question's keywords appear in the answer
  quality_score     — average of the two above
"""
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class EvaluationResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    quality_score: float
    evaluator: str


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens, stop-words removed."""
    _STOP = {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "of",
        "and", "or", "but", "for", "with", "that", "this", "are", "was",
        "be", "by", "from", "as", "not", "what", "how", "does", "do",
        "can", "which", "when", "where", "who", "will", "has", "have",
    }
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP]


def _overlap(a: list[str], b: list[str]) -> float:
    """F1 token overlap between two token lists."""
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    precision = common / len(b)
    recall = common / len(a)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _seq_sim(a: str, b: str) -> float:
    """SequenceMatcher ratio between two strings."""
    return round(SequenceMatcher(None, a.lower(), b.lower()).ratio(), 4)


class RAGASEvaluator:
    """
    Reference-free quality evaluator — no GPU, no model downloads.
    Uses token F1 overlap and sequence similarity as proxies for
    faithfulness and answer relevancy.
    """

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        # embed_model param kept for API compatibility but not used
        print("[eval] Lightweight text-overlap evaluator ready (no model required).")

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> EvaluationResult:
        ans_tok = _tokens(answer)
        q_tok = _tokens(question)

        # faithfulness: how much of the answer is supported by any context chunk
        if contexts:
            ctx_all = " ".join(contexts)
            ctx_tok = _tokens(ctx_all)
            faithfulness = _overlap(ans_tok, ctx_tok)
            # also try seq similarity for a blended score
            seq_faith = max(_seq_sim(answer[:400], c[:400]) for c in contexts)
            faithfulness = round((faithfulness + seq_faith) / 2, 4)
        else:
            faithfulness = 0.0

        # answer relevancy: how many question keywords appear in the answer
        answer_relevancy = _overlap(q_tok, ans_tok)
        # blend with sequence sim
        seq_rel = _seq_sim(question, answer[:200])
        answer_relevancy = round((answer_relevancy + seq_rel) / 2, 4)

        quality_score = round((faithfulness + answer_relevancy) / 2, 4)

        return EvaluationResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=0.0,
            quality_score=quality_score,
            evaluator="text-overlap",
        )
