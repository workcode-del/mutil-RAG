from __future__ import annotations

import argparse

from paper_rag.evaluation.comparison import DEFAULT_METRICS, save_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paper-rag evaluation reports")
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--output", default="outputs/evaluation_comparison.csv")
    args = parser.parse_args()

    target, table = save_comparison(args.reports, args.output, args.metrics)
    print(table)
    print(f"Saved comparison: {target}")


if __name__ == "__main__":
    main()
