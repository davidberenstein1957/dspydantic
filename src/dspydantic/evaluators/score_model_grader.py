"""Score model grader evaluator using LLM for numeric scoring."""

import json
import re
from typing import Any

import dspy


class ScoreModelGrader:
    """Evaluator that uses an LLM to assign a numeric score.

    Config options:
        criteria (str): Scoring criteria/prompt for the LLM
        lm (dspy.LM | None): Custom LM instance (default: uses optimizer's LM)
        temperature (float): Temperature for LLM (default: 0.0)
        system_prompt (str | None): Custom system prompt for judge
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize ScoreModelGrader.

        Args:
            config: Configuration dictionary with criteria, lm, temperature, system_prompt options.
        """
        self.config = config
        self.criteria = config.get("criteria", "Rate the quality on a scale of 0-1")
        self.lm = config.get("lm")
        self.temperature = config.get("temperature", 0.0)
        self.system_prompt = config.get("system_prompt")

    def evaluate(
        self,
        extracted: Any,
        expected: Any,
        input_data: dict[str, Any] | None = None,
        field_path: str | None = None,
    ) -> float:
        """Evaluate using LLM-based scoring.

        Args:
            extracted: Extracted value.
            expected: Expected value.
            input_data: Optional input data for context.
            field_path: Optional field path for context.

        Returns:
            Score between 0.0 and 1.0.
        """
        if self.lm is None:
            # Use default LM from dspy settings
            lm = dspy.settings.lm
            if lm is None:
                raise ValueError("No LM available for ScoreModelGrader")
        else:
            lm = self.lm

        # Build prompt
        prompt_parts = []
        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}")

        prompt_parts.append(f"Criteria: {self.criteria}")
        prompt_parts.append(f"\nExpected value: {expected}")
        prompt_parts.append(f"Extracted value: {extracted}")

        if field_path:
            prompt_parts.append(f"\nField: {field_path}")

        if input_data:
            prompt_parts.append(f"\nInput context: {input_data}")

        prompt_parts.append(
            "\nRespond with a JSON object containing a 'score' field (float between 0.0 and 1.0) "
            "and optionally a 'reasoning' field explaining your evaluation."
        )

        prompt = "\n\n".join(prompt_parts)

        # Use DSPy's ChainOfThought
        signature = "prompt -> evaluation"
        judge = dspy.ChainOfThought(signature)
        result = judge(prompt=prompt)

        # Extract evaluation from result
        evaluation_text = str(result.evaluation) if hasattr(result, "evaluation") else str(result)

        # Try to parse JSON from evaluation
        try:
            evaluation = json.loads(evaluation_text)
            score = float(evaluation.get("score", 0.5))
        except (json.JSONDecodeError, ValueError, AttributeError):
            # Try to extract score from text using regex
            score_match = re.search(r'"score"\s*:\s*([0-9.]+)', evaluation_text)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except ValueError:
                    score = 0.5
            else:
                # Fallback: try to find a number between 0 and 1
                score_match = re.search(r"\b(0\.\d+|1\.0|1)\b", evaluation_text)
                if score_match:
                    try:
                        score = float(score_match.group(1))
                    except ValueError:
                        score = 0.5
                else:
                    score = 0.5

        return max(0.0, min(1.0, score))
