"""Shared scoring utilities for evaluation functions."""

from typing import Any

try:
    from deepdiff import DeepDiff
except ImportError:
    DeepDiff = None  # type: ignore[assignment, misc]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_nested_value(data: dict[str, Any], path: str) -> Any:
    """Get value from nested dictionary using dot notation path."""
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def compare_values(extracted: Any, expected: Any, metric: str = "exact") -> float:
    """Compare extracted value with expected value.

    Handles nested structures including dictionaries and lists using DeepDiff.

    Args:
        extracted: Extracted value.
        expected: Expected value.
        metric: Comparison metric ("exact" or "levenshtein").

    Returns:
        Score between 0.0 and 1.0.
    """
    if DeepDiff is None:
        raise ImportError(
            "deepdiff is required for scoring. Install it with: pip install deepdiff"
        )

    # Check if we have nested structures (dict or list)
    has_nested_structures = isinstance(expected, dict | list) or isinstance(
        extracted, dict | list
    )

    # For nested structures, use DeepDiff's deep_distance for accurate comparison
    if has_nested_structures:
        diff = DeepDiff(
            expected,
            extracted,
            ignore_order=False,
            verbose_level=0,
            get_deep_distance=True,
        )
        # If diff is empty, structures are identical
        if not diff:
            return 1.0

        deep_distance = diff.get("deep_distance", 1.0)

        # For exact metric, return binary result (1.0 if identical, 0.0 otherwise)
        if metric == "exact":
            return 1.0 if deep_distance == 0.0 else 0.0

        # For levenshtein metric, use deep_distance as similarity score
        # DeepDiff's deep_distance is between 0 (identical) and 1 (very different)
        # Convert to similarity score: 1 - distance
        similarity = 1.0 - deep_distance
        return max(0.0, min(1.0, similarity))

    # For primitive types (non-nested), handle based on metric
    if metric == "exact":
        # Exact matching for primitives
        return 1.0 if extracted == expected else 0.0

    # For levenshtein metric with primitives, use Levenshtein distance
    str_extracted = str(extracted).strip()
    str_expected = str(expected).strip()

    if str_extracted == str_expected:
        return 1.0

    max_len = max(len(str_extracted), len(str_expected))
    if max_len == 0:
        return 1.0

    distance = levenshtein_distance(str_extracted, str_expected)
    similarity = 1.0 - (distance / max_len)
    return max(0.0, similarity)
