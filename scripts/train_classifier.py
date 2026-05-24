"""Stage C: train confidence classifiers for a given retrieval method.

Joins data/signals_{method}.json with data/results_{method}.json by question id,
fits a small bench of models (majority-vote dummy, decision tree, L2 + L1
logistic regression) each with 5-fold stratified CV and a *small* grid search
over hyperparameters. The grids are deliberately tiny so the "best" pick isn't
just CV-noise on ~80 samples.

The headline question is: does any model beat the majority-class baseline by a
meaningful margin? We report each model's CV accuracy / balanced accuracy / F1
and let the summary script rank them across methods.

Usage:
    uv run scripts/train_classifier.py                  # method=baseline
    uv run scripts/train_classifier.py --method graph_v1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

# sklearn 1.8 emits a FutureWarning per fit for LogisticRegression(penalty=...)
# nudging users toward the new `l1_ratio` API. With grid search × 5-fold CV we
# get hundreds of identical lines; silence them so real output stays readable.
warnings.filterwarnings(
    "ignore",
    message=".*penalty.*was deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Inconsistent values: penalty=.*",
    category=UserWarning,
)

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, DEFAULT_METHOD
from signals import feature_names_for, to_row


def paths_for(method: str) -> dict[str, Path]:
    return {
        "signals": DATA_DIR / f"signals_{method}.json",
        "results": DATA_DIR / f"results_{method}.json",
        "classifier": DATA_DIR / f"classifier_{method}.pkl",
        "report": DATA_DIR / f"classifier_{method}_report.json",
    }


def load_dataset(
    paths: dict[str, Path],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if not paths["signals"].exists():
        raise SystemExit(f"missing {paths['signals']} — run extract_signals.py --method ... first")
    if not paths["results"].exists():
        raise SystemExit(f"missing {paths['results']} — run evaluate.py --method ... first")

    signals = {s["id"]: s for s in json.loads(paths["signals"].read_text(encoding="utf-8"))}
    results = {r["id"]: r for r in json.loads(paths["results"].read_text(encoding="utf-8"))}

    common = sorted(signals.keys() & results.keys())
    if not common:
        raise SystemExit("no overlapping ids between signals.json and results.json")

    only_sig = sorted(signals.keys() - results.keys())
    only_res = sorted(results.keys() - signals.keys())
    if only_sig:
        print(f"[load] {len(only_sig)} ids in signals but not results (likely: eval was sampled)")
    if only_res:
        print(f"[load] {len(only_res)} ids in results but not signals (signals rebuild needed?)")

    X = np.array([to_row(signals[i], feature_names) for i in common], dtype=np.float32)
    y = np.array([1 if results[i]["passed"] else 0 for i in common], dtype=np.int32)
    tiers = [signals[i].get("tier", "?") for i in common]
    return X, y, common, tiers


# --- Model bench -----------------------------------------------------------
# Grids are intentionally tiny. With ~80 samples and 5-fold CV, a large grid
# would let us cherry-pick a lucky combo and overstate generalisation. Keep
# each grid in the ~6–24 range and use balanced_accuracy as the selector so
# class imbalance doesn't tilt the choice toward "always predict pass".

def build_models() -> list[tuple[str, object, dict]]:
    tree_grid = {
        "max_depth": [2, 3, 4],
        "min_samples_leaf": [4, 8],
        "class_weight": [None, "balanced"],
        "ccp_alpha": [0.0, 0.005],
    }  # 24 combos
    lr_l2_grid = {
        "lr__C": [0.1, 1.0, 10.0],
        "lr__class_weight": [None, "balanced"],
    }  # 6 combos
    lr_l1_grid = {
        "lr__C": [0.1, 1.0, 10.0],
        "lr__class_weight": [None, "balanced"],
    }  # 6 combos

    return [
        (
            "dummy_majority",
            DummyClassifier(strategy="most_frequent"),
            {},
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(random_state=42),
            tree_grid,
        ),
        (
            "logreg_l2",
            Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(
                    penalty="l2", solver="lbfgs", max_iter=2000, random_state=42,
                )),
            ]),
            lr_l2_grid,
        ),
        (
            "logreg_l1",
            Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(
                    penalty="l1", solver="liblinear", max_iter=2000, random_state=42,
                )),
            ]),
            lr_l1_grid,
        ),
    ]


def feature_importance_for(
    name: str,
    estimator: object,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """Return (feature, importance) pairs in `feature_names` order.

    Tree: Gini importances directly.
    LogReg pipelines: |coef_| (features are standardised, so magnitudes are comparable).
    Dummy: all zeros.
    """
    if isinstance(estimator, DecisionTreeClassifier):
        importances = estimator.feature_importances_
    elif isinstance(estimator, Pipeline) and "lr" in estimator.named_steps:
        coef = np.asarray(estimator.named_steps["lr"].coef_).reshape(-1)
        importances = np.abs(coef)
    else:
        importances = np.zeros(len(feature_names), dtype=np.float64)
    return list(zip(feature_names, [float(v) for v in importances]))


def fit_and_score(
    name: str,
    estimator: object,
    grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    baseline: float,
    feature_names: list[str],
) -> dict:
    """Fit `estimator` (with grid search if grid is non-empty) and report metrics."""
    if grid:
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X, y)
        best = search.best_estimator_
        best_params = {k: v for k, v in search.best_params_.items()}
        grid_best_bal_acc = float(search.best_score_)
        grid_size = len(search.cv_results_["params"])
    else:
        best = estimator
        best.fit(X, y)
        best_params = {}
        grid_best_bal_acc = None
        grid_size = 1

    cv_scores = cross_validate(
        best,
        X,
        y,
        cv=cv,
        scoring={"acc": "accuracy", "bal_acc": "balanced_accuracy", "f1": "f1"},
        n_jobs=-1,
    )
    cv_mean_acc = float(cv_scores["test_acc"].mean())
    cv_std_acc = float(cv_scores["test_acc"].std())
    cv_mean_bal = float(cv_scores["test_bal_acc"].mean())
    cv_std_bal = float(cv_scores["test_bal_acc"].std())
    cv_mean_f1 = float(cv_scores["test_f1"].mean())
    cv_std_f1 = float(cv_scores["test_f1"].std())

    importances = feature_importance_for(name, best, feature_names)
    importances_sorted = sorted(importances, key=lambda kv: kv[1], reverse=True)
    nonzero = [(n, v) for n, v in importances_sorted if v > 0.0]

    tree_text = None
    if isinstance(best, DecisionTreeClassifier):
        tree_text = export_text(best, feature_names=feature_names, decimals=3)

    return {
        "name": name,
        "estimator": best,
        "grid_size": grid_size,
        "best_params": best_params,
        "grid_best_balanced_accuracy": grid_best_bal_acc,
        "cv_mean_accuracy": cv_mean_acc,
        "cv_std_accuracy": cv_std_acc,
        "cv_mean_balanced_accuracy": cv_mean_bal,
        "cv_std_balanced_accuracy": cv_std_bal,
        "cv_mean_f1": cv_mean_f1,
        "cv_std_f1": cv_std_f1,
        "cv_improvement_over_baseline": cv_mean_acc - baseline,
        "feature_importances": [{"name": n, "importance": v} for n, v in importances_sorted],
        "nonzero_feature_count": len(nonzero),
        "nonzero_features": [{"name": n, "importance": v} for n, v in nonzero],
        "tree_text": tree_text,
    }


def print_model_block(res: dict, baseline: float, n_splits: int) -> None:
    print(f"\n[model] {res['name']}  (grid={res['grid_size']})")
    if res["best_params"]:
        print(f"  best params : {res['best_params']}")
    if res["grid_best_balanced_accuracy"] is not None:
        print(f"  grid best bal_acc : {res['grid_best_balanced_accuracy']:.3f}")
    print(
        f"  {n_splits}-fold CV  acc={res['cv_mean_accuracy']:.3f}±{res['cv_std_accuracy']:.3f}  "
        f"bal_acc={res['cv_mean_balanced_accuracy']:.3f}±{res['cv_std_balanced_accuracy']:.3f}  "
        f"f1={res['cv_mean_f1']:.3f}±{res['cv_std_f1']:.3f}"
    )
    print(
        f"  vs baseline acc={baseline:.3f}  delta={res['cv_improvement_over_baseline']:+.3f}"
    )
    top = res["nonzero_features"][:8]
    if top:
        print("  top features:")
        for f in top:
            print(f"    {f['importance']:.3f}  {f['name']}")
    if res["tree_text"]:
        print("  tree:")
        for line in res["tree_text"].splitlines():
            print(f"    {line}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--method", default=DEFAULT_METHOD,
        help=f"retrieval method name (default: {DEFAULT_METHOD})",
    )
    args = ap.parse_args()
    paths = paths_for(args.method)
    feature_names = feature_names_for(args.method)
    print(f"[train] method={args.method}")
    print(f"[train] signals  ← {paths['signals'].name}")
    print(f"[train] results  ← {paths['results'].name}")
    print(f"[train] feature set ({len(feature_names)} cols)")

    X, y, ids, tiers = load_dataset(paths, feature_names)
    n_pass = int(y.sum())
    n_fail = len(y) - n_pass
    baseline = max(n_pass, n_fail) / len(y)
    print(f"[train] {len(ids)} samples ({n_pass} PASS, {n_fail} FAIL)")
    print(f"[train] majority-class baseline accuracy: {baseline:.3f}")

    n_splits = min(5, n_pass, n_fail)
    if n_splits < 2:
        raise SystemExit(
            f"[train] too few samples in one class for CV (n_pass={n_pass}, n_fail={n_fail})"
        )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results: list[dict] = []
    for name, estimator, grid in build_models():
        res = fit_and_score(name, estimator, grid, X, y, cv, baseline, feature_names)
        results.append(res)
        print_model_block(res, baseline, n_splits)

    # Rank by balanced accuracy (same metric used by the grid search), tiebreak
    # on accuracy. Dummy is always last unless something is very wrong.
    results.sort(
        key=lambda r: (r["cv_mean_balanced_accuracy"], r["cv_mean_accuracy"]),
        reverse=True,
    )
    best = results[0]
    print(f"\n[train] best model by CV balanced_accuracy: {best['name']} "
          f"({best['cv_mean_balanced_accuracy']:.3f})")

    # Persist the best model so a predict-time script has a stable artefact.
    paths["classifier"].parent.mkdir(parents=True, exist_ok=True)
    with paths["classifier"].open("wb") as f:
        pickle.dump(
            {
                "model": best["estimator"],
                "model_name": best["name"],
                "feature_names": feature_names,
                "method": args.method,
            },
            f,
        )
    print(f"[save] {paths['classifier']}  ({best['name']})")

    # Strip non-JSON-friendly fields before serialising.
    def to_report_model(r: dict) -> dict:
        out = {k: v for k, v in r.items() if k != "estimator"}
        return out

    report = {
        "method": args.method,
        "n_samples": len(ids),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "baseline_accuracy": baseline,
        "cv_n_splits": n_splits,
        "feature_names": feature_names,
        "best_model": best["name"],
        "models": [to_report_model(r) for r in results],
    }
    with paths["report"].open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[save] {paths['report']}")


if __name__ == "__main__":
    main()
