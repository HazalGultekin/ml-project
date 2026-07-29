"""
Output Parsing (Pipeline Step 6).

Parses the raw model responses in raw_extractions.csv into structured
entity lists (companies, models, technologies) and writes them to
parsed_extractions.csv.

Two response formats need to be handled:
    - structured: a JSON object, optionally wrapped in a ```json fence.
    - zero_shot / few_shot: free text with "Companies:", "AI Models:",
      and "AI Technologies:" sections, each followed by a bullet list.

Parsing failures (invalid JSON, missing sections) are recorded rather
than silently dropped, so Step 8 (Error Analysis) can quantify them
per model/strategy.

Usage:
    python parse_outputs.py
"""

import json
import re

import pandas as pd

INPUT_PATH = "raw_extractions.csv"
OUTPUT_PATH = "parsed_extractions.csv"

# --- Structured (JSON) parsing -------------------------------------------

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _as_str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [str(v).strip() for v in value if str(v).strip()]
    return [v for v in items if not NEGATION_RE.search(v)]


def parse_structured(raw_output: str) -> tuple[list[str], list[str], list[str], str]:
    """Returns (companies, models, technologies, parse_error)."""
    text = raw_output.strip()

    fence_match = JSON_FENCE_RE.search(text)
    if fence_match:
        json_text = fence_match.group(1)
    else:
        # Some responses prepend commentary (e.g. "Here is the JSON:") with no
        # code fence. Fall back to the outermost {...} span.
        start, end = text.find("{"), text.rfind("}")
        json_text = text[start:end + 1] if start != -1 and end > start else text

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return [], [], [], f"invalid_json: {e}"

    if not isinstance(data, dict):
        return [], [], [], "invalid_json: top-level value is not an object"

    companies = _as_str_list(data.get("companies", []))
    models = _as_str_list(data.get("models", []))
    technologies = _as_str_list(data.get("technologies", []))
    return companies, models, technologies, ""


# --- Zero-shot / Few-shot (labeled list) parsing --------------------------

HEADER_RE = re.compile(
    r"(?im)^\s*\**\s*(companies|ai models|models|ai technologies|technologies)\s*:?\s*\**\s*:?\s*$"
)

HEADER_TO_CATEGORY = {
    "companies": "companies",
    "ai models": "models",
    "models": "models",
    "ai technologies": "technologies",
    "technologies": "technologies",
}

BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+?)\s*$")

EMPTY_MARKERS = {"none", "n/a", "none.", "none explicitly mentioned", "none explicitly mentioned in the article", "empty"}

# Some models (llama3 especially) write self-negating bullet items instead of
# omitting the entity, e.g. "Retrieval-Augmented Generation (RAG) (not
# mentioned in the article)". Left as-is, these would be parsed as positive
# predictions and inflate false positives at evaluation time (Step 7), even
# though the model is explicitly saying the entity is NOT present.
NEGATION_RE = re.compile(
    r"\bnot\s+(?:explicitly\s+)?(?:mentioned|named|present|included)\b"
    r"|\bno\s+specific\b"
    r"|\bdoes(?:n't| not)\s+mention\b"
    r"|\bisn't\s+mentioned\b"
    r"|\bremoved\s+it\s+from\s+the\s+list\b",
    re.IGNORECASE,
)


def _extract_bullets(block: str) -> list[str]:
    items = []
    for line in block.splitlines():
        line = line.strip().rstrip(".")
        if not line:
            continue
        match = BULLET_RE.match(line)
        item = match.group(1).strip() if match else line
        item = item.strip("*").strip()
        if not item or item.lower() in EMPTY_MARKERS:
            continue
        if "none explicitly mentioned" in item.lower():
            continue
        if NEGATION_RE.search(item):
            continue
        items.append(item)
    return items


def parse_labeled_list(raw_output: str) -> tuple[list[str], list[str], list[str], str]:
    text = raw_output.strip()

    headers = list(HEADER_RE.finditer(text))
    if not headers:
        return [], [], [], "no_sections_found"

    sections = {"companies": [], "models": [], "technologies": []}
    for i, h in enumerate(headers):
        category = HEADER_TO_CATEGORY[h.group(1).lower()]
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections[category].extend(_extract_bullets(text[start:end]))

    missing = [k for k, v in sections.items() if not v and k not in _sections_explicitly_empty(text, headers)]
    parse_error = f"missing_sections: {missing}" if missing else ""

    return sections["companies"], sections["models"], sections["technologies"], parse_error


def _sections_explicitly_empty(text: str, headers) -> set[str]:
    """A section is 'explicitly empty' if its header exists but yields no bullets
    because the model wrote e.g. 'None'. We don't want to flag those as parse
    failures, only sections whose header is missing entirely."""
    found = set()
    for h in headers:
        found.add(HEADER_TO_CATEGORY[h.group(1).lower()])
    return found


# --- Main ------------------------------------------------------------------

def main():
    df = pd.read_csv(INPUT_PATH)

    rows = []
    for _, row in df.iterrows():
        raw_output = "" if pd.isna(row["raw_output"]) else str(row["raw_output"])

        if row["error"] and not pd.isna(row["error"]):
            companies, models, technologies, parse_error = [], [], [], "api_error"
        elif row["strategy"] == "structured":
            companies, models, technologies, parse_error = parse_structured(raw_output)
        else:
            companies, models, technologies, parse_error = parse_labeled_list(raw_output)

        rows.append({
            "article_id": row["article_id"],
            "model": row["model"],
            "strategy": row["strategy"],
            "companies": "; ".join(companies),
            "models": "; ".join(models),
            "technologies": "; ".join(technologies),
            "parse_success": parse_error == "",
            "parse_error": parse_error,
        })

    parsed_df = pd.DataFrame(rows)
    parsed_df.to_csv(OUTPUT_PATH, index=False)

    total = len(parsed_df)
    failures = (~parsed_df["parse_success"]).sum()
    print(f"Parsed {total} rows. Failures: {failures} ({failures / total:.1%}).")
    if failures:
        print("\nFailures by model/strategy:")
        print(parsed_df[~parsed_df["parse_success"]].groupby(["model", "strategy"]).size())


if __name__ == "__main__":
    main()
