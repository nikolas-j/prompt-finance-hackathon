"""Aggregate classifier reports across methods and models into one table.

Reads every data/classifier_*_report.json (each contains a `models: [...]`
list) and prints a single ranked table with one row per (method, model). The
ranking goal is the same as the training goal: find the configuration that
beats the per-method majority-class baseline by the largest balanced-accuracy
margin. Pure JSON compiler — no fitting, no recomputation.

Usage:
    uv run scripts/summary.py
"""
from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_signed(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def method_name_from_report(report_path: Path) -> str:
    name = report_path.name
    if not (name.startswith("classifier_") and name.endswith("_report.json")):
        return name
    return name[len("classifier_") : -len("_report.json")]


def collect_rows(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    report_paths = sorted(data_dir.glob("classifier_*_report.json"))
    for report_path in report_paths:
        method = method_name_from_report(report_path)
        report = load_json(report_path)
        if not isinstance(report, dict):
            continue

        baseline = report.get("baseline_accuracy")
        best_model_name = report.get("best_model")

        results_path = data_dir / f"results_{method}.json"
        pass_rate = None
        if results_path.exists():
            results = load_json(results_path)
            if isinstance(results, list) and results:
                n_pass = sum(1 for r in results if r.get("passed"))
                pass_rate = n_pass / len(results)

        models = report.get("models")
        # Back-compat: older reports stored a single model flat in the report.
        if not models:
            models = [{
                "name": "model",
                "best_params": report.get("best_params"),
                "grid_best_balanced_accuracy": report.get("grid_best_balanced_accuracy"),
                "cv_mean_accuracy": report.get("cv_mean_accuracy"),
                "cv_std_accuracy": report.get("cv_std_accuracy"),
                "cv_mean_balanced_accuracy": report.get("cv_mean_balanced_accuracy"),
                "cv_std_balanced_accuracy": report.get("cv_std_balanced_accuracy"),
                "cv_mean_f1": report.get("cv_mean_f1"),
                "cv_improvement_over_baseline": report.get("cv_improvement_over_baseline"),
                "nonzero_feature_count": report.get("nonzero_feature_count"),
                "feature_importances": report.get("feature_importances", []),
            }]

        for m in models:
            feature_importances = m.get("feature_importances", [])
            nonzero = [f for f in feature_importances if float(f.get("importance", 0.0)) > 0.0]
            top_names = [f.get("name", "?") for f in nonzero[:5]]
            nonzero_count = m.get("nonzero_feature_count")
            if nonzero_count is None:
                nonzero_count = len(nonzero)

            delta_bal = None
            cv_bal = m.get("cv_mean_balanced_accuracy")
            if cv_bal is not None:
                # 0.5 is the majority-baseline balanced accuracy by definition.
                delta_bal = float(cv_bal) - 0.5

            rows.append(
                {
                    "method": method,
                    "model": m.get("name", "?"),
                    "is_best": m.get("name") == best_model_name,
                    "n": report.get("n_samples"),
                    "pass_rate": pass_rate,
                    "baseline": baseline,
                    "cv_acc": m.get("cv_mean_accuracy"),
                    "cv_bal_acc": cv_bal,
                    "cv_f1": m.get("cv_mean_f1"),
                    "delta_acc": m.get("cv_improvement_over_baseline"),
                    "delta_bal": delta_bal,
                    "nonzero_count": nonzero_count,
                    "top_features": top_names,
                    "best_params": m.get("best_params"),
                }
            )
    return rows


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("No classifier report files found in data/.")
        return

    rows.sort(
        key=lambda r: (
            r["cv_bal_acc"] is None,
            -(r["cv_bal_acc"] or -1.0),
            -(r["cv_acc"] or -1.0),
        )
    )

    line = "=" * 132
    print(line)
    print("TAXXA CONFIDENCE CLASSIFIER SUMMARY  (one row per method × model; best per method marked *)")
    print(line)
    header = (
        f"{'method':16} {'model':16} {'n':>4} {'pass':>6} {'base':>6} "
        f"{'cv_acc':>8} {'cv_bal':>8} {'cv_f1':>7} {'Δacc':>8} {'Δbal':>8} {'nz':>4}"
    )
    print(header)
    print("-" * 132)

    for row in rows:
        star = "*" if row["is_best"] else " "
        print(
            f"{str(row['method']):16} "
            f"{star}{str(row['model']):15} "
            f"{str(row['n'] if row['n'] is not None else '-'):>4} "
            f"{fmt_float(row['pass_rate'], 2):>6} "
            f"{fmt_float(row['baseline'], 2):>6} "
            f"{fmt_float(row['cv_acc']):>8} "
            f"{fmt_float(row['cv_bal_acc']):>8} "
            f"{fmt_float(row['cv_f1']):>7} "
            f"{fmt_signed(row['delta_acc']):>8} "
            f"{fmt_signed(row['delta_bal']):>8} "
            f"{str(row['nonzero_count'] if row['nonzero_count'] is not None else '-'):>4}"
        )
        top = ", ".join(row["top_features"]) if row["top_features"] else "-"
        print(f"    top features: {top}")
        if row.get("best_params"):
            print(f"    best params : {row['best_params']}")
        print("-" * 132)

    # A short verdict line so the headline is hard to miss.
    best_overall = rows[0]
    print(
        "\nBest overall (by CV balanced accuracy): "
        f"{best_overall['method']} / {best_overall['model']}  "
        f"bal_acc={fmt_float(best_overall['cv_bal_acc'])}  "
        f"Δbal={fmt_signed(best_overall['delta_bal'])}  "
        f"acc={fmt_float(best_overall['cv_acc'])}  "
        f"Δacc={fmt_signed(best_overall['delta_acc'])}"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    rows = collect_rows(data_dir)
    print_summary(rows)


if __name__ == "__main__":
    main()
