"""Optimize extraction using a criteria-based judge and no labeled examples.

Examples have input only (expected_output=None). Uses the ScoreJudge evaluator
([Configure Evaluators](https://davidberenstein1957.github.io/dspydantic/guides/evaluators/configure/))
with criteria to score each extraction; the optimizer iteratively improves
prompts/descriptions to maximize that score.
"""

from typing import Any, Literal

import dspy
from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer, ScoreJudge


class ReviewSummary(BaseModel):
    """Extract sentiment and brief summary from a review."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    summary: str = Field(description="One-sentence summary of the main points")


JUDGE_CRITERIA = """
Score the extraction (0.0–1.0) using these criteria:
1. Sentiment must reflect the overall tone of the input (positive/negative/neutral).
2. Summary must be one concise sentence and capture key points without inventing facts.
3. Output must be valid JSON matching the schema (sentiment, summary).
4. Score 0.5 for minor issues; below 0.5 for wrong sentiment or irrelevant summary.
Respond with JSON: {"score": <float>, "reasoning": "<brief>"}.
"""


def main() -> None:
    """Run optimization with judge-only evaluation and no labeled examples."""
    # Unlabeled examples: input only, expected_output=None
    examples = [
        Example(
            text="The film was dull and the actors seemed bored. Save your money.",
            expected_output=None,
        ),
        Example(
            text="Best movie of the year. Stunning visuals and a touching story.",
            expected_output=None,
        ),
        Example(
            text="It was okay. Nice soundtrack but the plot was predictable.",
            expected_output=None,
        ),
    ]

    # Use ScoreJudge from evaluator config (same as docs)
    score_judge = ScoreJudge(config={"criteria": JUDGE_CRITERIA, "temperature": 0.0})

    def judge_fn(
        example: Example,
        extracted_data: dict[str, Any],
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        return score_judge.evaluate(
            extracted=extracted_data,
            expected=None,
            input_data=example.input_data,
        )

    # Configure DSPy before running
    lm = dspy.LM("openai/gpt-4o-mini")  # use env OPENAI_API_KEY
    dspy.configure(lm=lm)

    optimizer = PydanticOptimizer(
        model=ReviewSummary,
        examples=examples,
        evaluate_fn=judge_fn,
        verbose=True,
        system_prompt="You extract sentiment and a brief summary from user reviews.",
        instruction_prompt="Extract sentiment and summary from the input text.",
    )

    result = optimizer.optimize()

    print("\n--- Results ---")
    print(f"Baseline score: {result.baseline_score:.2%}")
    print(f"Optimized score: {result.optimized_score:.2%}")
    print(f"Improvement: {result.metrics.get('improvement', 0):+.2%}")
    print("\nOptimized system prompt:", result.optimized_system_prompt or "(none)")
    print("Optimized instruction:", result.optimized_instruction_prompt or "(none)")
    print("Optimized descriptions:")
    for k, v in (result.optimized_descriptions or {}).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
