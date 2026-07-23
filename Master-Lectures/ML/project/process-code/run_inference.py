"""
Model Inference (Pipeline Step 5).

Runs every evaluation article through every prompting strategy
(zero_shot, few_shot, structured) on every model (Llama 3.1 8B Instruct,
Gemma 3 12B IT), via the OpenRouter API, and stores the raw text
responses in raw_extractions.csv.

Usage:
    python run_inference.py --limit 5        # smoke test on 5 articles
    python run_inference.py                  # full run (100 articles)

Requires OPENROUTER_API_KEY to be set (e.g. in a local .env file, which
is gitignored).

The script is resumable: it skips (article_id, model, strategy) triples
that are already present in raw_extractions.csv, so an interrupted run
can simply be re-launched.
"""

import argparse
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from prompt_templates import PROMPT_BUILDERS, build_prompt

load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ.get("OPENROUTER_API_KEY")

MODELS = {
    "llama3": "meta-llama/llama-3.1-8b-instruct",
    "gemma3": "google/gemma-3-12b-it",
}

EVAL_ARTICLES_PATH = "evaluation_articles.csv"
OUTPUT_PATH = "raw_extractions.csv"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 120


def call_model(model_id: str, prompt: str) -> tuple[str, str]:
    """Call the OpenRouter chat completions API. Returns (raw_output, error)."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content, ""
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return "", last_error


def load_existing_results() -> pd.DataFrame:
    if os.path.exists(OUTPUT_PATH):
        return pd.read_csv(OUTPUT_PATH)
    return pd.DataFrame(
        columns=["article_id", "model", "strategy", "raw_output", "error"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N articles (for smoke-testing before the full run).",
    )
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Add it to a local .env file "
            "(OPENROUTER_API_KEY=sk-or-...) before running."
        )

    eval_df = pd.read_csv(EVAL_ARTICLES_PATH)
    if args.limit:
        eval_df = eval_df.head(args.limit)

    existing = load_existing_results()
    done_keys = set(
        zip(existing["article_id"], existing["model"], existing["strategy"])
    )

    total = len(eval_df) * len(MODELS) * len(PROMPT_BUILDERS)
    done = len(done_keys)
    print(f"Total combinations: {total}. Already done: {done}.")

    results = existing.to_dict("records")

    for _, row in eval_df.iterrows():
        article_id = row["article_id"]
        body = row["body"]

        for model_key, model_id in MODELS.items():
            for strategy in PROMPT_BUILDERS:
                key = (article_id, model_key, strategy)
                if key in done_keys:
                    continue

                prompt = build_prompt(strategy, body)
                raw_output, error = call_model(model_id, prompt)

                results.append(
                    {
                        "article_id": article_id,
                        "model": model_key,
                        "strategy": strategy,
                        "raw_output": raw_output,
                        "error": error,
                    }
                )
                done_keys.add(key)
                done += 1

                status = "OK" if not error else f"ERROR: {error[:80]}"
                print(f"[{done}/{total}] {article_id} | {model_key} | {strategy} -> {status}")

                # Persist after every call so a crash never loses progress.
                pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone. Results saved to {OUTPUT_PATH} ({len(results)} rows).")


if __name__ == "__main__":
    main()
