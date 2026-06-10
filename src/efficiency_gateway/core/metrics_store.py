import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4


@dataclass
class RunRecord:
    run_id: str
    question: str
    status: str
    cache_hit: int              # 0 or 1 (SQLite has no bool)
    selected_model: str
    input_tokens: int
    output_tokens: int
    baseline_input_tokens: int
    estimated_cost: float
    baseline_cost: float
    tokens_saved: int
    cost_saved: float
    token_reduction_pct: float
    cost_reduction_pct: float
    latency_ms: float
    quality_score: float
    faithfulness: float
    answer_relevancy: float
    evaluator: str
    timestamp: float


def new_run_id() -> str:
    return str(uuid4())


class MetricsStore:
    _CREATE = """
        CREATE TABLE IF NOT EXISTS runs (
            run_id              TEXT PRIMARY KEY,
            question            TEXT,
            status              TEXT,
            cache_hit           INTEGER,
            selected_model      TEXT,
            input_tokens        INTEGER,
            output_tokens       INTEGER,
            baseline_input_tokens INTEGER,
            estimated_cost      REAL,
            baseline_cost       REAL,
            tokens_saved        INTEGER,
            cost_saved          REAL,
            token_reduction_pct REAL,
            cost_reduction_pct  REAL,
            latency_ms          REAL,
            quality_score       REAL,
            faithfulness        REAL,
            answer_relevancy    REAL,
            evaluator           TEXT,
            timestamp           REAL
        )
    """

    def __init__(self, db_path: str = "data/metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(self._CREATE)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, record: RunRecord) -> None:
        placeholders = ", ".join(f":{k}" for k in asdict(record))
        sql = f"INSERT OR REPLACE INTO runs VALUES ({placeholders})"
        with self._conn() as c:
            c.execute(sql, asdict(record))

    def fetch_all(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM runs ORDER BY timestamp DESC").fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        sql = """
            SELECT
                COUNT(*)                        AS total_runs,
                SUM(cache_hit)                  AS cache_hits,
                ROUND(AVG(tokens_saved), 1)     AS avg_tokens_saved,
                ROUND(AVG(cost_saved), 8)       AS avg_cost_saved,
                ROUND(AVG(token_reduction_pct), 2) AS avg_token_reduction_pct,
                ROUND(AVG(cost_reduction_pct), 2)  AS avg_cost_reduction_pct,
                ROUND(AVG(latency_ms), 1)       AS avg_latency_ms,
                ROUND(AVG(quality_score), 4)    AS avg_quality_score
            FROM runs
        """
        with self._conn() as c:
            row = c.execute(sql).fetchone()
        return dict(row) if row else {}
