import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficiency_gateway.core.cost_tracker import CostTracker, calculate_savings


tracker = CostTracker()

baseline = tracker.estimate_from_counts(
    model="gpt-4o-mini",
    input_tokens=12_000,
    output_tokens=1_200,
)

optimized = tracker.estimate_from_counts(
    model="gpt-4o-mini",
    input_tokens=3_800,
    output_tokens=900,
)

savings = calculate_savings(baseline, optimized)

print("Baseline:", baseline)
print("Optimized:", optimized)
print("Savings:", savings)