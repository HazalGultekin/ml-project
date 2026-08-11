"""
Error Analysis (Pipeline Step 8).

Goes through every (article, model, strategy) prediction and records
individual error instances, grouped by LLM and prompting strategy:

    - missed_entity:            a ground-truth entity the model did not predict
    - entity_confusion:         a predicted entity that IS in the ground truth,
                                 but under a different category (e.g. predicted
                                 as a company when the ground truth has it
                                 labeled as a model)
    - false_positive_entity:    a predicted entity that is not in the ground
                                 truth under any category (hallucinated / not
                                 actually mentioned)
    - json_formatting_failure:  a structured-prompt response that failed to
                                 parse as valid JSON (from Step 6)

Reuses the same normalization as evaluate.py so error counts here are
consistent with the precision/recall/F1 numbers from Step 7.

Usage:
    python error_analysis.py
"""

import pandas as pd

from evaluate import CATEGORIES, normalize, to_entity_set

PARSED_PATH = "parsed_extractions.csv"
GROUND_TRUTH_PATH = "ground_truth_annotations.csv"
OUTPUT_PATH = "error_analysis.csv"


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

    records = []

    for _, row in parsed.iterrows():
        article_id, model, strategy = row["article_id"], row["model"], row["strategy"]
        gt_entities = gt_sets[article_id]

        if not row["parse_success"]:
            records.append({
                "model": model, "strategy": strategy, "article_id": article_id,
                "error_type": "json_formatting_failure", "category": "all",
                "entity": "", "detail": row["parse_error"],
            })
            # Fall through rather than `continue`: the predicted columns are
            # empty for a failed parse, so every ground-truth entity for this
            # article is correctly recorded as missed below, matching the
            # FN counts in evaluate.py.

        # Map normalized entity -> original category, for confusion lookups.
        gt_norm_to_category = {}
        for cat in CATEGORIES:
            for entity in gt_entities[cat]:
                gt_norm_to_category.setdefault(entity, cat)

        for cat in CATEGORIES:
            predicted = to_entity_set(row[cat])
            actual = gt_entities[cat]

            for entity in actual - predicted:
                records.append({
                    "model": model, "strategy": strategy, "article_id": article_id,
                    "error_type": "missed_entity", "category": cat,
                    "entity": entity, "detail": "",
                })

            for entity in predicted - actual:
                other_category = gt_norm_to_category.get(entity)
                if other_category and other_category != cat:
                    records.append({
                        "model": model, "strategy": strategy, "article_id": article_id,
                        "error_type": "entity_confusion", "category": cat,
                        "entity": entity, "detail": f"ground truth lists it under '{other_category}'",
                    })
                else:
                    records.append({
                        "model": model, "strategy": strategy, "article_id": article_id,
                        "error_type": "false_positive_entity", "category": cat,
                        "entity": entity, "detail": "",
                    })

    error_df = pd.DataFrame(records)
    error_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(error_df)} error instances to {OUTPUT_PATH}\n")
    summary = error_df.groupby(["model", "strategy", "error_type"]).size().unstack(fill_value=0)
    pd.set_option("display.width", 140)
    print(summary)


if __name__ == "__main__":
    main()
