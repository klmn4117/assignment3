"""Re-evaluate assignment 2 predictions and verify every metric with scikit-learn."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ["positive", "negative"]
CATEGORIES = {"implicit_positive", "negative_word_scope", "target_ambiguity"}


def load_predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "text", "gold", "rule", "llm"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("CSV requires id,text,gold,rule,llm columns")
        rows = list(reader)
    if not rows or any(None in row or any(not row.get(key, "").strip() for key in required)
                       for row in rows):
        raise ValueError("Prediction rows must be nonempty and complete")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    if any(row[key] not in LABELS for row in rows for key in ("gold", "rule", "llm")):
        raise ValueError("Invalid label")
    return rows


def ratio(a, b):
    return a / b if b else None


def calculate(gold, predicted):
    if not gold or len(gold) != len(predicted):
        raise ValueError("Nonempty equal-length label lists required")
    if any(x not in LABELS for x in [*gold, *predicted]):
        raise ValueError("Invalid label")
    # Same orientation as the lecture: rows=prediction, columns=gold.
    matrix = [[sum(g == gl and p == pl for g, p in zip(gold, predicted))
               for gl in LABELS] for pl in LABELS]
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        fp = sum(matrix[i]) - tp
        fn = sum(row[i] for row in matrix) - tp
        per_class[label] = {
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "support": tp + fn}
    return {"labels": LABELS, "matrix_rows_prediction_columns_gold": matrix,
            "tp": matrix[0][0], "fp": matrix[0][1],
            "fn": matrix[1][0], "tn": matrix[1][1],
            "accuracy": sum(matrix[i][i] for i in range(2)) / len(gold),
            "per_class": per_class}


def verify_sklearn(gold, predicted, manual):
    import numpy as np
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    # sklearn uses rows=gold, columns=prediction; transpose before comparison.
    matrix = confusion_matrix(gold, predicted, labels=LABELS).T.tolist()
    if matrix != manual["matrix_rows_prediction_columns_gold"]:
        raise ValueError("Confusion matrix differs from scikit-learn")
    if not math.isclose(accuracy_score(gold, predicted), manual["accuracy"], abs_tol=1e-12):
        raise ValueError("Accuracy differs from scikit-learn")
    p, r, f, support = precision_recall_fscore_support(
        gold, predicted, labels=LABELS, average=None, zero_division=np.nan)
    for i, label in enumerate(LABELS):
        for key, expected in [("precision", p[i]), ("recall", r[i]), ("f1", f[i])]:
            value = manual["per_class"][label][key]
            if not ((value is None and np.isnan(expected)) or
                    (value is not None and math.isclose(value, expected, abs_tol=1e-12))):
                raise ValueError(f"{label} {key} differs from scikit-learn")
        if manual["per_class"][label]["support"] != int(support[i]):
            raise ValueError("Support differs from scikit-learn")


def validate_annotations(rows, annotations):
    expected = {(method, row["id"]) for row in rows for method in ("rule", "llm")
                if row[method] != row["gold"]}
    actual = [(a["method"], a["id"]) for a in annotations]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Annotations must cover every error exactly once, and no correct rows")
    if any(a["category"] not in CATEGORIES or not a["explanation"].strip() for a in annotations):
        raise ValueError("Invalid category or missing explanation")


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/predictions.csv")
    parser.add_argument("--annotations", type=Path, help="Optional error category CSV")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    import sklearn
    rows = load_predictions(args.data)
    gold = [row["gold"] for row in rows]
    metrics = {}
    for method in ("rule", "llm"):
        predicted = [row[method] for row in rows]
        metrics[method] = calculate(gold, predicted)
        verify_sklearn(gold, predicted, metrics[method])
        metrics[method]["error_ids"] = [r["id"] for r in rows if r[method] != r["gold"]]
    annotation_path = args.annotations
    if annotation_path is None and args.data.resolve() == (ROOT / "data/predictions.csv").resolve():
        annotation_path = ROOT / "data/error_annotations.csv"
    annotations = []
    if annotation_path is not None:
        with annotation_path.open(encoding="utf-8-sig", newline="") as handle:
            annotations = list(csv.DictReader(handle))
        validate_annotations(rows, annotations)
    counts = Counter((a["method"], a["category"]) for a in annotations)
    summary = [{"category": cat, "rule": counts["rule", cat], "llm": counts["llm", cat]}
               for cat in sorted(CATEGORIES)]
    source_paths = [args.data] + ([annotation_path] if annotation_path is not None else [])
    output = {"total": len(rows), "positive_class": "positive",
              "undefined_metric": None, "methods": metrics,
              "verification": {"library": "scikit-learn", "version": sklearn.__version__,
                               "python": platform.python_version(), "all_matched": True},
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source_paths},
              "new_llm_inference": False, "error_counts": summary if annotation_path else None}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evaluation.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metric_rows = [{"method": method, "class": label, "accuracy": m["accuracy"], **values}
                   for method, m in metrics.items() for label, values in m["per_class"].items()]
    write_csv(args.out / "metrics.csv", metric_rows,
              ["method", "class", "accuracy", "precision", "recall", "f1", "support"])
    matrix_rows = [{"method": method, "prediction": label,
                    "gold_positive": m["matrix_rows_prediction_columns_gold"][i][0],
                    "gold_negative": m["matrix_rows_prediction_columns_gold"][i][1]}
                   for method, m in metrics.items() for i, label in enumerate(LABELS)]
    write_csv(args.out / "confusion_matrices.csv", matrix_rows,
              ["method", "prediction", "gold_positive", "gold_negative"])
    notes = {(a["method"], a["id"]): a for a in annotations}
    errors = [{"method": method, "id": row["id"],
               "category": notes.get((method, row["id"]), {}).get("category", ""),
               "explanation": notes.get((method, row["id"]), {}).get("explanation", ""),
               "gold": row["gold"], "prediction": row[method], "text": row["text"]}
              for row in rows for method in ("rule", "llm") if row[method] != row["gold"]]
    write_csv(args.out / "errors.csv", errors,
              ["method", "id", "category", "explanation", "gold", "prediction", "text"])
    write_csv(args.out / "error_categories.csv", summary if annotation_path else [],
              ["category", "rule", "llm"])
    print(json.dumps({"verification": output["verification"], "methods": metrics},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
