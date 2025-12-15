"""Utilities for extracting and applying descriptions from GLiNER2 schemas."""

from typing import Any


def extract_gliner_descriptions(schema: dict[str, Any]) -> dict[str, str]:
    """Extract descriptions from a GLiNER2 schema.

    GLiNER2 schemas can contain:
    - entities: list[str] or dict[str, str] mapping entity names to descriptions
    - relations: list[str] or dict[str, str] mapping relation names to descriptions
    - classifications: dict with "labels" (list[str] or dict[str, str]) and optional "multi_label"

    Args:
        schema: GLiNER2 schema dictionary.

    Returns:
        Dictionary mapping schema element paths to their descriptions.
        Paths use dot notation: "entities.{name}", "relations.{name}", "classifications.{label}"

    Example:
        ```python
        schema = {
            "entities": {"person": "Names of people", "location": "Places"},
            "relations": ["works_for", "located_in"],
            "classifications": {
                "labels": {"sentiment": "positive or negative", "category": "topic category"}
            }
        }
        descriptions = extract_gliner_descriptions(schema)
        # Returns: {
        #     "entities.person": "Names of people",
        #     "entities.location": "Places",
        #     "relations.works_for": "works_for",
        #     "relations.located_in": "located_in",
        #     "classifications.sentiment": "positive or negative",
        #     "classifications.category": "topic category"
        # }
        ```
    """
    descriptions: dict[str, str] = {}

    # Extract entity descriptions
    if "entities" in schema:
        entities = schema["entities"]
        if isinstance(entities, dict):
            for entity_name, description in entities.items():
                descriptions[f"entities.{entity_name}"] = (
                    description if isinstance(description, str) else str(description)
                )
        elif isinstance(entities, list):
            for entity_name in entities:
                if isinstance(entity_name, str):
                    descriptions[f"entities.{entity_name}"] = entity_name

    # Extract relation descriptions
    if "relations" in schema:
        relations = schema["relations"]
        if isinstance(relations, dict):
            for relation_name, description in relations.items():
                descriptions[f"relations.{relation_name}"] = (
                    description if isinstance(description, str) else str(description)
                )
        elif isinstance(relations, list):
            for relation_name in relations:
                if isinstance(relation_name, str):
                    descriptions[f"relations.{relation_name}"] = relation_name

    # Extract classification descriptions
    # Handle multiple formats:
    # 1. {"classifications": {"labels": {...}}} - single classification
    # 2. {"classifications": {"task_name": {"labels": [...]}}} - multiple classifications
    if "classifications" in schema:
        classifications = schema["classifications"]
        if isinstance(classifications, dict):
            # Check if it's format 1: direct "labels" key
            if "labels" in classifications:
                labels = classifications["labels"]
                if isinstance(labels, dict):
                    for label_name, description in labels.items():
                        descriptions[f"classifications.{label_name}"] = (
                            description if isinstance(description, str) else str(description)
                        )
                elif isinstance(labels, list):
                    for label_name in labels:
                        if isinstance(label_name, str):
                            descriptions[f"classifications.{label_name}"] = label_name
            else:
                # Format 2: each key is a classification task name
                for task_name, task_config in classifications.items():
                    if isinstance(task_config, dict):
                        # Check if there's a description for the task itself
                        task_description = task_config.get("description")
                        if task_description:
                            descriptions[f"classifications.{task_name}"] = task_description
                        else:
                            # No description - auto-generate from task name
                            descriptions[f"classifications.{task_name}"] = task_name
                    elif isinstance(task_config, list):
                        # Simple format: task_name -> list of labels
                        descriptions[f"classifications.{task_name}"] = task_name

    # Auto-generate descriptions for fields without descriptions
    # This ensures we always have something to optimize
    auto_generated: dict[str, str] = {}

    # Entities: if list format, generate descriptions from names
    if "entities" in schema:
        entities = schema["entities"]
        if isinstance(entities, list):
            for entity_name in entities:
                if isinstance(entity_name, str):
                    path = f"entities.{entity_name}"
                    if path not in descriptions:
                        auto_generated[path] = entity_name

    # Relations: if list format, generate descriptions from names
    if "relations" in schema:
        relations = schema["relations"]
        if isinstance(relations, list):
            for relation_name in relations:
                if isinstance(relation_name, str):
                    path = f"relations.{relation_name}"
                    if path not in descriptions:
                        auto_generated[path] = relation_name

    # Classifications: generate descriptions for task names if not already present
    if "classifications" in schema:
        classifications = schema["classifications"]
        if isinstance(classifications, dict):
            # Format 2: each key is a classification task
            if "labels" not in classifications:
                for task_name in classifications.keys():
                    path = f"classifications.{task_name}"
                    if path not in descriptions:
                        auto_generated[path] = task_name

    # Merge auto-generated descriptions
    descriptions.update(auto_generated)

    return descriptions


def apply_optimized_gliner_descriptions(
    schema: dict[str, Any], optimized_descriptions: dict[str, str]
) -> dict[str, Any]:
    """Apply optimized descriptions back to a GLiNER2 schema.

    Args:
        schema: Original GLiNER2 schema dictionary.
        optimized_descriptions: Dictionary mapping schema element paths to optimized descriptions.

    Returns:
        Modified GLiNER2 schema dictionary with optimized descriptions applied.

    Example:
        ```python
        schema = {
            "entities": {"person": "Names of people"},
            "relations": ["works_for"]
        }
        optimized = {
            "entities.person": "Full names of individuals",
            "relations.works_for": "Employment relationships"
        }
        updated_schema = apply_optimized_gliner_descriptions(schema, optimized)
        # Returns: {
        #     "entities": {"person": "Full names of individuals"},
        #     "relations": {"works_for": "Employment relationships"}
        # }
        ```
    """
    import copy

    # Create a deep copy to avoid modifying the original
    updated_schema = copy.deepcopy(schema)

    # Update entity descriptions
    if "entities" in updated_schema:
        entities = updated_schema["entities"]
        if isinstance(entities, dict):
            for entity_name in list(entities.keys()):
                path = f"entities.{entity_name}"
                if path in optimized_descriptions:
                    entities[entity_name] = optimized_descriptions[path]
        elif isinstance(entities, list):
            # Convert list to dict if we have optimized descriptions
            entities_dict: dict[str, str] = {}
            for entity_name in entities:
                if isinstance(entity_name, str):
                    path = f"entities.{entity_name}"
                    if path in optimized_descriptions:
                        entities_dict[entity_name] = optimized_descriptions[path]
                    else:
                        entities_dict[entity_name] = entity_name
            if entities_dict:
                updated_schema["entities"] = entities_dict

    # Update relation descriptions
    if "relations" in updated_schema:
        relations = updated_schema["relations"]
        if isinstance(relations, dict):
            for relation_name in list(relations.keys()):
                path = f"relations.{relation_name}"
                if path in optimized_descriptions:
                    relations[relation_name] = optimized_descriptions[path]
        elif isinstance(relations, list):
            # Convert list to dict if we have optimized descriptions
            relations_dict: dict[str, str] = {}
            for relation_name in relations:
                if isinstance(relation_name, str):
                    path = f"relations.{relation_name}"
                    if path in optimized_descriptions:
                        relations_dict[relation_name] = optimized_descriptions[path]
                    else:
                        relations_dict[relation_name] = relation_name
            if relations_dict:
                updated_schema["relations"] = relations_dict

    # Update classification descriptions
    # Handle multiple formats:
    # 1. {"classifications": {"labels": {...}}} - single classification
    # 2. {"classifications": {"task_name": {"labels": [...]}}} - multiple classifications
    if "classifications" in updated_schema:
        classifications = updated_schema["classifications"]
        if isinstance(classifications, dict):
            if "labels" in classifications:
                # Format 1: direct "labels" key
                labels = classifications["labels"]
                if isinstance(labels, dict):
                    for label_name in list(labels.keys()):
                        path = f"classifications.{label_name}"
                        if path in optimized_descriptions:
                            labels[label_name] = optimized_descriptions[path]
                elif isinstance(labels, list):
                    # Convert list to dict if we have optimized descriptions
                    labels_dict: dict[str, str] = {}
                    for label_name in labels:
                        if isinstance(label_name, str):
                            path = f"classifications.{label_name}"
                            if path in optimized_descriptions:
                                labels_dict[label_name] = optimized_descriptions[path]
                            else:
                                labels_dict[label_name] = label_name
                    if labels_dict:
                        classifications["labels"] = labels_dict
            else:
                # Format 2: each key is a classification task name
                for task_name, task_config in classifications.items():
                    if isinstance(task_config, dict):
                        path = f"classifications.{task_name}"
                        if path in optimized_descriptions:
                            # Add or update the description field
                            task_config["description"] = optimized_descriptions[path]
                    elif isinstance(task_config, list):
                        # Simple format: task_name -> list of labels
                        # Convert to dict format with description
                        path = f"classifications.{task_name}"
                        if path in optimized_descriptions:
                            classifications[task_name] = {
                                "labels": task_config,
                                "description": optimized_descriptions[path],
                            }

    return updated_schema
