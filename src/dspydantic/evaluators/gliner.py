"""Evaluation functions for GLiNER2 schema optimization."""

from collections.abc import Callable
from typing import Any

from dspydantic.evaluators._scoring import compare_values
from dspydantic.types import Example


def _normalize_expected_output(expected: dict[str, Any]) -> dict[str, Any]:
    """Normalize expected output to match the format of normalized GLiNER2 results.

    Converts relation tuples to dicts and ensures consistent structure.

    Args:
        expected: Expected output dictionary.

    Returns:
        Normalized expected output dictionary.
    """
    normalized: dict[str, Any] = {}

    # Copy entities as-is (they're already in the right format)
    if "entities" in expected:
        normalized["entities"] = expected["entities"]

    # Normalize relations: convert tuples to dicts
    if "relations" in expected:
        relations = expected["relations"]
        normalized["relations"] = {}
        for rel_name, rel_tuples in relations.items():
            if isinstance(rel_tuples, list):
                # Convert list of tuples to list of dicts
                normalized["relations"][rel_name] = [
                    (
                        {"subject": tup[0], "object": tup[1]}
                        if isinstance(tup, tuple) and len(tup) == 2
                        else tup
                    )
                    for tup in rel_tuples
                ]
            else:
                normalized["relations"][rel_name] = rel_tuples

    # Copy other keys (classifications, etc.)
    for key, value in expected.items():
        if key not in ("entities", "relations"):
            normalized[key] = value

    return normalized


def _normalize_gliner_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize GLiNER2 extraction result to a standard format.

    GLiNER2 can return results in different formats depending on what was extracted:
    - Entities: {"entities": {"person": ["John"], "location": ["NYC"]}}
    - Relations: {"relation_extraction": {"works_for": [("John", "Company")]}}
    - Classifications: {"sentiment": "positive", "category": "tech"}

    This function normalizes all formats to a consistent structure.

    Args:
        result: Raw GLiNER2 extraction result.

    Returns:
        Normalized result dictionary.
    """
    normalized: dict[str, Any] = {}

    # Handle entities
    if "entities" in result:
        normalized["entities"] = result["entities"]

    # Handle relations
    if "relation_extraction" in result:
        # Convert relation tuples to a more standard format
        relations = result["relation_extraction"]
        normalized["relations"] = {}
        for rel_name, rel_tuples in relations.items():
            if isinstance(rel_tuples, list):
                # Convert list of tuples to list of dicts
                normalized["relations"][rel_name] = [
                    (
                        {"subject": tup[0], "object": tup[1]}
                        if isinstance(tup, tuple) and len(tup) == 2
                        else tup
                    )
                    for tup in rel_tuples
                ]
            else:
                normalized["relations"][rel_name] = rel_tuples

    # Handle classifications (direct keys)
    for key, value in result.items():
        if key not in ("entities", "relation_extraction"):
            normalized[key] = value

    return normalized


def _compare_entities(
    extracted: dict[str, list[str]], expected: dict[str, list[str]], metric: str = "exact"
) -> float:
    """Compare entity extraction results.

    Compares entity lists with order independence (set comparison).
    Each entity type is scored independently, then averaged.

    Args:
        extracted: Extracted entities dict.
        expected: Expected entities dict.
        metric: Comparison metric ("exact" or "levenshtein").

    Returns:
        Score between 0.0 and 1.0.
    """
    all_entity_types = set(extracted.keys()) | set(expected.keys())
    if not all_entity_types:
        return 1.0

    entity_scores = []
    for entity_type in all_entity_types:
        extracted_list = extracted.get(entity_type, [])
        expected_list = expected.get(entity_type, [])

        # Convert to sets for order-independent comparison
        extracted_set = set(extracted_list)
        expected_set = set(expected_list)

        if metric == "exact":
            # Exact match: all entities must match exactly
            if extracted_set == expected_set:
                entity_scores.append(1.0)
            else:
                # Partial credit: intersection / union (Jaccard similarity)
                intersection = len(extracted_set & expected_set)
                union = len(extracted_set | expected_set)
                if union == 0:
                    entity_scores.append(1.0)
                else:
                    entity_scores.append(intersection / union)
        else:
            # Levenshtein metric: use Jaccard similarity
            intersection = len(extracted_set & expected_set)
            union = len(extracted_set | expected_set)
            if union == 0:
                entity_scores.append(1.0)
            else:
                entity_scores.append(intersection / union)

    return sum(entity_scores) / len(entity_scores) if entity_scores else 0.0


def _compare_relations(
    extracted: dict[str, list[dict[str, str]]],
    expected: dict[str, list[dict[str, str]]],
    metric: str = "exact",
) -> float:
    """Compare relation extraction results.

    Compares relation lists with order independence.
    Each relation type is scored independently, then averaged.

    Args:
        extracted: Extracted relations dict (normalized to dicts).
        expected: Expected relations dict (normalized to dicts).
        metric: Comparison metric ("exact" or "levenshtein").

    Returns:
        Score between 0.0 and 1.0.
    """
    all_relation_types = set(extracted.keys()) | set(expected.keys())
    if not all_relation_types:
        return 1.0

    relation_scores = []
    for relation_type in all_relation_types:
        extracted_list = extracted.get(relation_type, [])
        expected_list = expected.get(relation_type, [])

        # Convert to sets of tuples for order-independent comparison
        # Handle both dict format {"subject": ..., "object": ...} and tuple format
        def normalize_relation(rel: dict[str, str] | tuple[str, str]) -> tuple[str, str]:
            """Normalize relation to tuple format."""
            if isinstance(rel, dict):
                return (rel.get("subject", ""), rel.get("object", ""))
            elif isinstance(rel, tuple) and len(rel) == 2:
                return rel
            else:
                return ("", "")

        extracted_set = {normalize_relation(rel) for rel in extracted_list}
        expected_set = {normalize_relation(rel) for rel in expected_list}

        if metric == "exact":
            # Exact match: all relations must match exactly
            if extracted_set == expected_set:
                relation_scores.append(1.0)
            else:
                # Partial credit: intersection / union (Jaccard similarity)
                intersection = len(extracted_set & expected_set)
                union = len(extracted_set | expected_set)
                if union == 0:
                    relation_scores.append(1.0)
                else:
                    relation_scores.append(intersection / union)
        else:
            # Levenshtein metric: use Jaccard similarity
            intersection = len(extracted_set & expected_set)
            union = len(extracted_set | expected_set)
            if union == 0:
                relation_scores.append(1.0)
            else:
                relation_scores.append(intersection / union)

    return sum(relation_scores) / len(relation_scores) if relation_scores else 0.0


def default_gliner_evaluate_fn(
    extractor: Any,  # GLiNER2 extractor instance
    schema: dict[str, Any],
    metric: str = "exact",
    exclude_fields: list[str] | None = None,
) -> Callable[[Example, dict[str, str], str | None, str | None], float]:
    """Create a default evaluation function that uses GLiNER2 for extraction.

    Args:
        extractor: GLiNER2 extractor instance
            (from GLiNER2.from_pretrained() or GLiNER2.from_api()).
        schema: GLiNER2 schema dictionary.
        metric: Comparison metric to use. Options:
            - "exact": Exact matching using DeepDiff with deep_distance
              for nested structures (default)
            - "levenshtein": Levenshtein distance-based matching for primitives,
              DeepDiff deep_distance for nested structures
        exclude_fields: Optional list of field paths to exclude from evaluation.
            Field paths use dot notation
            (e.g., ["entities.person", "relations.works_for"]).

    Returns:
        An evaluation function that performs extraction using GLiNER2 and compares
        with expected output.
    """
    def evaluate(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        """Default evaluation function using GLiNER2 for extraction.

        Args:
            example: The example with input_data and expected_output.
            optimized_descriptions: Dictionary of optimized schema element descriptions.
            optimized_system_prompt: Optimized system prompt
                (not used for GLiNER2, but kept for API compatibility).
            optimized_instruction_prompt: Optimized instruction prompt
                (not used for GLiNER2, but kept for API compatibility).

        Returns:
            Score between 0.0 and 1.0 based on extraction accuracy.
        """
        # Get input data from example
        input_data = example.input_data

        # Handle Pydantic models for input_data
        if isinstance(input_data, dict):
            input_text = input_data.get("text")
        else:
            input_text = str(input_data)

        if not input_text:
            return 0.0

        # Apply optimized descriptions to the schema
        from dspydantic.extractors.gliner import apply_optimized_gliner_descriptions

        optimized_schema = apply_optimized_gliner_descriptions(schema, optimized_descriptions)

        # Extract using GLiNER2 with optimized schema
        try:
            # Build GLiNER2 schema from optimized_schema dict
            gliner_schema = _build_gliner_schema(extractor, optimized_schema)

            # Use GLiNER2's extract method with the schema
            result = extractor.extract(input_text, gliner_schema)
        except Exception:
            # If extraction fails, return 0.0
            return 0.0

        # Get expected output
        expected_output = example.expected_output
        if expected_output is None:
            # If no expected output, return 0.0 (can't evaluate)
            return 0.0

        # Convert expected_output to dict if it's a Pydantic model
        if hasattr(expected_output, "model_dump"):
            expected_output = expected_output.model_dump()

        # Normalize both expected and extracted results to a common format for comparison
        normalized_result = _normalize_gliner_result(result)
        # Function is defined at module level above, mypy false positive
        normalized_expected = _normalize_expected_output(expected_output)  # type: ignore[no-redef]

        # Calculate score using shared scoring utilities
        return _calculate_gliner_score(
            normalized_expected, normalized_result, metric=metric, exclude_fields=exclude_fields
        )

    return evaluate


def _build_gliner_schema(extractor: Any, schema_dict: dict[str, Any]) -> Any:
    """Build a GLiNER2 schema object from a dictionary.

    Args:
        extractor: GLiNER2 extractor instance.
        schema_dict: Dictionary representation of GLiNER2 schema.

    Returns:
        GLiNER2 schema object.
    """
    # Use GLiNER2's schema builder
    gliner_schema = extractor.create_schema()

    # Add entities
    if "entities" in schema_dict:
        entities = schema_dict["entities"]
        if isinstance(entities, dict):
            gliner_schema = gliner_schema.entities(entities)
        elif isinstance(entities, list):
            gliner_schema = gliner_schema.entities(entities)

    # Add relations
    if "relations" in schema_dict:
        relations = schema_dict["relations"]
        if isinstance(relations, dict):
            gliner_schema = gliner_schema.relations(relations)
        elif isinstance(relations, list):
            gliner_schema = gliner_schema.relations(relations)

    # Add classifications
    # GLiNER2 supports multiple classification tasks via multiple .classification() calls
    # Each classification can be specified as a separate key in the schema
    # Format: {"classifications": {"sentiment": {"labels": [...], "multi_label": bool}}}
    if "classifications" in schema_dict:
        classifications = schema_dict["classifications"]
        if isinstance(classifications, dict):
            # Handle multiple classification tasks
            for label_name, label_config in classifications.items():
                if isinstance(label_config, dict):
                    labels = label_config.get("labels", [])
                    multi_label = label_config.get("multi_label", False)
                    cls_threshold = label_config.get("cls_threshold", 0.5)

                    if isinstance(labels, list | dict):
                        gliner_schema = gliner_schema.classification(
                            label_name,
                            labels,
                            multi_label=multi_label,
                            cls_threshold=cls_threshold,
                        )
                elif isinstance(label_config, list | dict):
                    # Simple format: just labels
                    gliner_schema = gliner_schema.classification(label_name, label_config)

    return gliner_schema


def _calculate_gliner_score(
    expected: dict[str, Any],
    extracted: dict[str, Any],
    metric: str = "exact",
    exclude_fields: list[str] | None = None,
) -> float:
    """Calculate a score between expected and extracted GLiNER2 data.

    Handles entities and relations with special comparison logic:
    - Entities: Compares lists with order independence (set comparison)
    - Relations: Compares lists of dicts/tuples with order independence

    Args:
        expected: Expected output dictionary (normalized).
        extracted: Extracted output dictionary (normalized).
        metric: Comparison metric ("exact" or "levenshtein").
        exclude_fields: Optional list of field paths to exclude from scoring.

    Returns:
        Score between 0.0 and 1.0.
    """
    # Get all keys from both dictionaries
    all_keys = set(expected.keys()) | set(extracted.keys())

    # Filter out excluded fields
    if exclude_fields:
        excluded_set = set(exclude_fields)
        all_keys = {
            key
            for key in all_keys
            if not any(key == excl or key.startswith(f"{excl}.") for excl in excluded_set)
        }

    if not all_keys:
        # Fallback to comparing entire structures
        return compare_values(extracted, expected, metric=metric)

    # Compare each field with special handling for entities and relations
    field_scores = []
    for key in all_keys:
        extracted_val = extracted.get(key)
        expected_val = expected.get(key)

        if extracted_val is None and expected_val is None:
            field_scores.append(1.0)
        elif extracted_val is None or expected_val is None:
            field_scores.append(0.0)
        else:
            # Special handling for entities and relations
            if key == "entities":
                field_score = _compare_entities(extracted_val, expected_val, metric=metric)
            elif key == "relations":
                field_score = _compare_relations(extracted_val, expected_val, metric=metric)
            else:
                # Use standard comparison for other fields
                field_score = compare_values(extracted_val, expected_val, metric=metric)
            field_scores.append(field_score)

    score = sum(field_scores) / len(field_scores) if field_scores else 0.0
    return max(0.0, min(1.0, score))


