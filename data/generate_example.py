"""Generate deterministic fictional weekly data, never actual DKS observations."""

import csv
import math
from datetime import date, timedelta
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    history = []
    future = []
    for i in range(116):
        day = date(2023, 1, 2) + timedelta(weeks=i)
        promo = int(i % 13 == 0)
        if i >= 104:
            future.append([day, promo])
            continue
        ticket = 75 + 0.08 * i + 4 * math.sin(2 * math.pi * i / 52) - 2 * promo
        transactions = 900 + 2 * i + 80 * math.sin(2 * math.pi * i / 52) + 130 * promo
        history.append(
            [
                day,
                round(ticket, 2),
                round(transactions),
                round(ticket * transactions, 2),
                round(transactions * 3.5),
                promo,
            ]
        )
    for name, header, rows in [
        (
            "example.csv",
            ["date", "ticket", "transactions", "revenue", "traffic", "promotion"],
            history,
        ),
        ("future.csv", ["date", "promotion"], future),
    ]:
        with (root / name).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    print("Generated fictional examples in data/.")


if __name__ == "__main__":
    main()
