"""Label model grader evaluator using LLM for categorical labeling."""

import json
from typing import Any

import dspy


class LabelModelGrader:
    """Evaluator that uses an LLM to select a categorical label.

    Config options:
        allowed_labels (list[str]): List of valid categorical labels
        lm (dspy.LM | None): Custom LM instance
        exact_match_score (float): Score when labels match exactly (default: 1.0)
        partial_match_score (float): Score for partial matches (default: 0.5)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize LabelModelGrader.

        Args:
            config: Configuration dictionary with allowed_labels, lm, exact_match_score,
                partial_match_score options.
        """
        self.config = config
        self.allowed_labels = config.get("allowed_labels", [])
        if not self.allowed_labels:
            raise ValueError("allowed_labels must be provided for LabelModelGrader")
        self.lm = config.get("lm")
        self.exact_match_score = config.get("exact_match_score", 1.0)
        self.partial_match_score = config.get("partial_match_score", 0.5)

    def evaluate(
        self,
        extracted: Any,
        expected: Any,
        input_data: dict[str, Any] | None = None,
        field_path: str | None = None,
    ) -> float:
        """Evaluate using LLM-based label selection.

        Args:
            extracted: Extracted value.
            expected: Expected value (should be one of allowed_labels).
            input_data: Optional input data for context.
            field_path: Optional field path for context.

        Returns:
            Score between 0.0 and 1.0.
        """
        if self.lm is None:
            # Use default LM from dspy settings
            lm = dspy.settings.lm
            if lm is None:
                raise ValueError("No LM available for LabelModelGrader")
        else:
            lm = self.lm

        # Convert to strings for comparison
        extracted_str = str(extracted).strip().lower()
        expected_str = str(expected).strip().lower()

        # Check for exact match first
        if extracted_str == expected_str:
            return self.exact_match_score

        # Check if expected is in allowed labels
        expected_lower = [label.lower() for label in self.allowed_labels]
        if expected_str not in expected_lower:
            # Expected label not in allowed labels - use LLM to determine match
            prompt_parts = []
            prompt_parts.append(
                f"Select the best matching label from: {', '.join(self.allowed_labels)}"
            )
            prompt_parts.append(f"\nExpected label: {expected}")
            prompt_parts.append(f"Extracted label: {extracted}")

            if field_path:
                prompt_parts.append(f"\nField: {field_path}")

            if input_data:
                prompt_parts.append(f"\nInput context: {input_data}")

            prompt_parts.append(
                "\nRespond with a JSON object containing a 'label' field (selected label) "
                "and optionally a 'reasoning' field."
            )

            prompt = "\n\n".join(prompt_parts)

            # Use DSPy's ChainOfThought
            signature = "prompt -> label_selection"
            grader = dspy.ChainOfThought(signature)
            result = grader(prompt=prompt)

            # Extract label from result
            label_text = str(result.label_selection) if hasattr(result, "label_selection") else str(result)

            # Try to parse JSON
            try:
                label_data = json.loads(label_text)
                selected_label = str(label_data.get("label", "")).strip().lower()
            except (json.JSONDecodeError, ValueError):
                # Try to find label in text
                selected_label = label_text.strip().lower()
                for label in self.allowed_labels:
                    if label.lower() in selected_label:
                        selected_label = label.lower()
                        break

            # Compare selected label with expected
            if selected_label == expected_str:
                return self.exact_match_score
            elif expected_str in selected_label or selected_label in expected_str:
                return self.partial_match_score
            else:
                return 0.0
        else:
            # Expected is in allowed labels, check if extracted matches
            extracted_lower = extracted_str.lower()
            if extracted_lower in expected_lower:
                idx = expected_lower.index(extracted_lower)
                if self.allowed_labels[idx].lower() == expected_str:
                    return self.exact_match_score

            # Check for partial match
            for label in self.allowed_labels:
                if expected_str in label.lower() or label.lower() in expected_str:
                    if extracted_lower in label.lower() or label.lower() in extracted_lower:
                        return self.partial_match_score

            return 0.0
