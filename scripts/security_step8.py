import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficiency_gateway.security.input_guard import InputSecurityGuard


def main():
    guard = InputSecurityGuard()

    text = input("Enter text to scan: ")

    result = guard.scan(text)

    print("\nSecurity result:")
    print(f"Valid: {result.is_valid}")
    print(f"Max risk score: {result.max_risk_score}")
    print(f"Prompt injection risk: {result.prompt_injection_risk}")
    print(f"PII risk: {result.pii_risk}")
    print(f"Secrets risk: {result.secrets_risk}")

    print("\nSanitized text:")
    print(result.sanitized_text)


if __name__ == "__main__":
    main()