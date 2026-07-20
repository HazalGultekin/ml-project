"""
Prompt templates for the information extraction task.

Three prompting strategies are provided, matching the project proposal:
    - Zero-shot
    - Few-shot
    - Structured (JSON output)

Each builder takes the raw article body text and returns a single prompt
string ready to be sent to an LLM (Llama 3, Gemma 3).
"""

# Technology labels are restricted to this predefined list during evaluation,
# per the proposal's "Information Extraction Targets" section.
TECHNOLOGY_LABELS = [
    "Retrieval-Augmented Generation (RAG)",
    "Reinforcement Learning",
    "Generative AI",
    "Multimodal AI",
    "AI Agents",
]

TASK_DESCRIPTION = (
    "Extract all company names, AI model names, and AI technologies "
    "mentioned in the article."
)

TECHNOLOGY_INSTRUCTION = (
    "Only consider the following AI technologies (use these exact labels, "
    "and only include a technology if it is explicitly mentioned in the "
    "article): " + ", ".join(TECHNOLOGY_LABELS) + "."
)

# ---------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------

def build_zero_shot_prompt(article_text: str) -> str:
    return (
        f"{TASK_DESCRIPTION}\n"
        f"{TECHNOLOGY_INSTRUCTION}\n\n"
        "Report the results as three labeled lists, in this exact format:\n"
        "Companies:\n- ...\n"
        "AI Models:\n- ...\n"
        "AI Technologies:\n- ...\n\n"
        "If a category has no entities, leave its list empty.\n\n"
        "Article:\n"
        f"{article_text}"
    )


# ---------------------------------------------------------------------------
# Few-shot
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "article": (
            "OpenAI announced GPT-5 while introducing new Retrieval-Augmented "
            "Generation capabilities. Anthropic also reported improvements to "
            "Claude."
        ),
        "output": (
            "Companies:\n- OpenAI\n- Anthropic\n"
            "AI Models:\n- GPT-5\n- Claude\n"
            "AI Technologies:\n- Retrieval-Augmented Generation (RAG)"
        ),
    },
    {
        "article": (
            "Google DeepMind unveiled Gemini's new multimodal reasoning "
            "capabilities, allowing the model to process images, audio, and "
            "text together. Meta's research team separately published work "
            "on reinforcement learning agents that can autonomously "
            "complete web browsing tasks."
        ),
        "output": (
            "Companies:\n- Google DeepMind\n- Meta\n"
            "AI Models:\n- Gemini\n"
            "AI Technologies:\n- Multimodal AI\n- Reinforcement Learning\n"
            "- AI Agents"
        ),
    },
]


def build_few_shot_prompt(article_text: str) -> str:
    examples_block = ""
    for i, example in enumerate(FEW_SHOT_EXAMPLES, start=1):
        examples_block += (
            f"Example {i}\n"
            f"Article:\n{example['article']}\n\n"
            f"Output:\n{example['output']}\n\n"
        )

    return (
        f"{TASK_DESCRIPTION}\n"
        f"{TECHNOLOGY_INSTRUCTION}\n\n"
        "Report the results as three labeled lists, in the same format as "
        "the examples below. If a category has no entities, leave its list "
        "empty.\n\n"
        f"{examples_block}"
        "Now extract the entities from the following article.\n\n"
        "Article:\n"
        f"{article_text}\n\n"
        "Output:"
    )


# ---------------------------------------------------------------------------
# Structured (JSON output)
# ---------------------------------------------------------------------------

JSON_SCHEMA_EXAMPLE = (
    '{\n'
    '  "companies": [],\n'
    '  "models": [],\n'
    '  "technologies": []\n'
    '}'
)


def build_structured_prompt(article_text: str) -> str:
    return (
        f"{TASK_DESCRIPTION}\n"
        f"{TECHNOLOGY_INSTRUCTION}\n\n"
        "Return ONLY a JSON object with this exact structure, and nothing "
        "else (no explanations, no markdown code fences):\n"
        f"{JSON_SCHEMA_EXAMPLE}\n\n"
        "If a category has no entities, return an empty list for it.\n\n"
        "Article:\n"
        f"{article_text}"
    )


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------

PROMPT_BUILDERS = {
    "zero_shot": build_zero_shot_prompt,
    "few_shot": build_few_shot_prompt,
    "structured": build_structured_prompt,
}


def build_prompt(strategy: str, article_text: str) -> str:
    """Build a prompt for the given strategy: 'zero_shot', 'few_shot', or 'structured'."""
    if strategy not in PROMPT_BUILDERS:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {list(PROMPT_BUILDERS)}."
        )
    return PROMPT_BUILDERS[strategy](article_text)


if __name__ == "__main__":
    sample_article = (
        "OpenAI announced GPT-5 while introducing new Retrieval-Augmented "
        "Generation capabilities. Anthropic also reported improvements to "
        "Claude."
    )
    for strategy in PROMPT_BUILDERS:
        print(f"\n{'=' * 20} {strategy} {'=' * 20}")
        print(build_prompt(strategy, sample_article))
