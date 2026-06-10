"""
Lightweight security guard — no llm_guard dependency.
Uses regex patterns to detect prompt injection, PII, and secrets.
Fast, reliable, and never crashes the process.
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardResult:
    original_text: str
    sanitized_text: str
    is_valid: bool
    max_risk_score: float
    prompt_injection_risk: float
    pii_risk: float
    secrets_risk: float


# --- Prompt injection patterns ---
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|context)",
    r"disregard\s+(previous|all|above|prior)\s+(instructions?|prompts?)",
    r"you\s+are\s+now\s+(a\s+)?(different|new|another|evil|dan)",
    r"jailbreak",
    r"act\s+as\s+if\s+you\s+(have\s+no|don.t\s+have)\s+(rules|restrictions|guidelines)",
    r"forget\s+(everything|all)\s+(you\s+)?(were\s+)?(told|trained|instructed)",
    r"do\s+anything\s+now",
    r"pretend\s+(you\s+are|to\s+be)\s+(not\s+an?\s+)?(ai|assistant|language model)",
    r"system\s*:\s*you\s+are",
    r"<\s*system\s*>",
    r"\[system\]",
    r"\\n\\nHuman:",
    r"\\n\\nAssistant:",
]

# --- PII patterns (for masking) ---
_PII_RULES: list[tuple[str, str, str]] = [
    ("EMAIL",       r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",     "[EMAIL]"),
    ("PHONE",       r"\b(\+?\d[\d\s\-().]{7,}\d)\b",                          "[PHONE]"),
    ("SSN",         r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",                       "[SSN]"),
    ("CREDIT_CARD", r"\b(?:\d[ -]?){13,16}\b",                                "[CREDIT_CARD]"),
    ("IP_ADDRESS",  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                           "[IP_ADDRESS]"),
]

# --- Secrets / API key patterns ---
_SECRETS_PATTERNS = [
    r"(?i)(api[_\s-]?key|secret[_\s-]?key|access[_\s-]?token|auth[_\s-]?token)\s*[=:]\s*\S+",
    r"(?i)(password|passwd|pwd)\s*[=:]\s*\S+",
    r"sk-[a-zA-Z0-9]{20,}",           # OpenAI-style key
    r"gsk_[a-zA-Z0-9]{20,}",          # Groq key
    r"AIza[0-9A-Za-z\-_]{35}",        # Google API key
    r"AKIA[0-9A-Z]{16}",              # AWS access key
    r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
]

_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_COMPILED_SECRETS   = [re.compile(p, re.IGNORECASE) for p in _SECRETS_PATTERNS]
_COMPILED_PII       = [(name, re.compile(pat), repl) for name, pat, repl in _PII_RULES]


class InputSecurityGuard:
    def __init__(self, prompt_injection_threshold: float = 0.5):
        self.threshold = prompt_injection_threshold
        print("[guard] Lightweight regex-based guard ready (injection + PII + secrets).")

    def scan(self, text: str) -> GuardResult:
        sanitized = text

        # 1. secrets check
        secrets_hits = sum(1 for p in _COMPILED_SECRETS if p.search(text))
        secrets_risk = min(secrets_hits * 0.5, 1.0)
        secrets_valid = secrets_risk < self.threshold

        # 2. PII masking
        pii_hits = 0
        for _name, pattern, replacement in _COMPILED_PII:
            new_text, n = pattern.subn(replacement, sanitized)
            if n:
                pii_hits += n
                sanitized = new_text
        pii_risk = min(pii_hits * 0.2, 1.0)
        pii_valid = True  # PII is masked, not blocked

        # 3. prompt injection check
        injection_hits = sum(1 for p in _COMPILED_INJECTION if p.search(text))
        injection_risk = min(injection_hits * 0.6, 1.0)
        injection_valid = injection_risk < self.threshold

        is_valid = secrets_valid and pii_valid and injection_valid
        max_risk = max(pii_risk, secrets_risk, injection_risk)

        return GuardResult(
            original_text=text,
            sanitized_text=sanitized,
            is_valid=is_valid,
            max_risk_score=max_risk,
            prompt_injection_risk=injection_risk,
            pii_risk=pii_risk,
            secrets_risk=secrets_risk,
        )

    def scan_documents(self, documents: list):
        safe, blocked = [], []
        for doc in documents:
            result = self.scan(doc.page_content)
            doc.metadata.update({
                "security_is_valid": result.is_valid,
                "security_max_risk_score": result.max_risk_score,
                "prompt_injection_risk": result.prompt_injection_risk,
                "pii_risk": result.pii_risk,
                "secrets_risk": result.secrets_risk,
            })
            if result.is_valid:
                doc.page_content = result.sanitized_text
                safe.append(doc)
            else:
                blocked.append(doc)
        return safe, blocked
