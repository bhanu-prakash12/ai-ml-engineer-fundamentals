"""Deterministic synthetic support-ticket dataset for the data-libraries notebooks.

Small, local, and reproducible (fixed seed) -- no external download required.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 42
N_ROWS = 300
CATEGORIES = ["billing", "login", "performance", "bug", None]  # None -> missing category
PRIORITIES = ["low", "medium", "high"]

OUT_PATH = Path(__file__).parent / "data" / "support_tickets.csv"


def generate_rows(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        category = rng.choice(CATEGORIES)
        row = {
            "ticket_id": i,
            "user_id": rng.randint(1, 50),          # many tickets per user -> good groupby demo
            "category": category,
            "priority": rng.choice(PRIORITIES),
            "resolved": rng.choice([True, False]),
            "response_time_minutes": round(rng.uniform(5, 500), 1),
        }
        rows.append(row)

    # inject deterministic duplicates to demonstrate dedup handling
    rows.extend(rows[:10])
    return rows


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(N_ROWS, SEED)
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
