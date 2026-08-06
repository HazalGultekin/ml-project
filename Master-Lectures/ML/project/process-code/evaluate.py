"""
Performance Evaluation (Pipeline Step 7).

Compares parsed_extractions.csv against ground_truth_annotations.csv and
computes Precision, Recall, and F1-score, reported separately per
(model, strategy, entity category), plus an overall row per (model,
strategy) combining all three categories.

Entity matching is done on a normalized string (lowercase, parenthetical
asides stripped, punctuation/whitespace collapsed) rather than the raw
text, since:
    - The technology labels used in ground truth ("Retrieval-Augmented
      Generation") and the labels used in the prompts ("Retrieval-Augmented
      Generation (RAG)") differ only by a parenthetical, which this
      normalization resolves without a special case.
    - Predicted companies/models are frequently followed by a clarifying
      aside, e.g. "MIT (Massachusetts Institute of Technology)".

This is a simplification: it does not handle every possible surface
variation (e.g. "Google DeepMind" vs. "DeepMind" as two different
mentions of arguably the same entity). Genuine mismatches of that kind
are left as real precision/recall errors for Step 8 to analyze, rather
than papered over here.

Metrics are micro-averaged: TP/FP/FN counts are summed across all 100
articles for a given (model, strategy, category) before computing
precision/recall/F1, rather than averaging per-article scores.

Usage:
    python evaluate.py
"""

import re

import pandas as pd

PARSED_PATH = "parsed_extractions.csv"
GROUND_TRUTH_PATH = "ground_truth_annotations.csv"
OUTPUT_PATH = "evaluation_metrics.csv"

CATEGORIES = ["companies", "models", "technologies"]

PAREN_RE = re.compile(r"\([^)]*\)")
PUNCT_RE = re.compile(r"[^\w\s&-]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize(entity: str) -> str:
    text = PAREN_RE.sub("", entity)
    text = text.lower()
    text = PUNCT_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def to_entity_set(cell) -> set[str]:
    if pd.isna(cell) or not str(cell).strip():
        return set()
    return {normalize(item) for item in str(cell).split(";") if normalize(item)}


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    parsed = pd.read_csv(PARSED_PATH)
    gt = pd.read_csv(GROUND_TRUTH_PATH)

    gt_sets = {
        row["article_id"]: {
            "companies": to_entity_set(row["gt_companies"]),
            "models": to_entity_set(row["gt_models"]),
            "technologies": to_entity_set(row["gt_technologies"]),
        }
        for _, row in gt.iterrows()
    }

    rows = []
    for (model, strategy), group in parsed.groupby(["model", "strategy"]):
        counts = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in CATEGORIES}

        for _, row in group.iterrows():
            gt_entities = gt_sets[row["article_id"]]
            for cat in CATEGORIES:
                predicted = to_entity_set(row[cat])
                actual = gt_entities[cat]

                counts[cat]["tp"] += len(predicted & actual)
                counts[cat]["fp"] += len(predicted - actual)
                counts[cat]["fn"] += len(actual - predicted)

        overall = {"tp": 0, "fp": 0, "fn": 0}
        for cat in CATEGORIES:
            c = counts[cat]
            precision, recall, f1 = prf1(c["tp"], c["fp"], c["fn"])
            rows.append({
                "model": model, "strategy": strategy, "category": cat,
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            })
            for k in overall:
                overall[k] += c[k]

        precision, recall, f1 = prf1(overall["tp"], overall["fp"], overall["fn"])
        rows.append({
            "model": model, "strategy": strategy, "category": "overall",
            "tp": overall["tp"], "fp": overall["fp"], "fn": overall["fn"],
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(metrics_df)} rows to {OUTPUT_PATH}\n")
    pd.set_option("display.width", 120)
    print(metrics_df[metrics_df["category"] == "overall"]
          .sort_values("f1", ascending=False)
          .to_string(index=False))


if __name__ == "__main__":
    main()
