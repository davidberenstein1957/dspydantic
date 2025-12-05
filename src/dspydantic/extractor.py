"""Utilities for extracting and applying field descriptions from Pydantic models."""

import copy
from typing import Any

from pydantic import BaseModel


def extract_field_descriptions(
    model: type[BaseModel], prefix: str = ""
) -> dict[str, str]:
    """Extract field descriptions from a Pydantic model recursively.
    
    Args:
        model: The Pydantic model class.
        prefix: Prefix for nested field paths (used internally for recursion).
    
    Returns:
        Dictionary mapping field paths to their descriptions.
        Field paths use dot notation for nested fields (e.g., "user.name").
    
    Example:
        ```python
        from pydantic import BaseModel, Field
        
        class User(BaseModel):
            name: str = Field(description="User's full name")
            age: int = Field(description="User's age")
        
        descriptions = extract_field_descriptions(User)
        # Returns: {"name": "User's full name", "age": "User's age"}
        ```
    """
    descriptions: dict[str, str] = {}
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})

    def resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve a $ref reference."""
        if ref.startswith("#/$defs/"):
            ref_name = ref.replace("#/$defs/", "")
            return defs.get(ref_name)
        return None

    def extract_from_schema(
        schema_dict: dict[str, Any], current_prefix: str = "", defs_dict: dict[str, Any] | None = None
    ) -> None:
        """Recursively extract descriptions from JSON schema."""
        if defs_dict is None:
            defs_dict = {}
        properties = schema_dict.get("properties", {})
        for field_name, field_schema in properties.items():
            field_path = (
                f"{current_prefix}.{field_name}" if current_prefix else field_name
            )

            # Handle $ref references (Pydantic v2 nested models)
            if "$ref" in field_schema:
                ref_schema = resolve_ref(field_schema["$ref"], defs_dict)
                if ref_schema:
                    # Extract description from the field itself if present
                    if "description" in field_schema:
                        descriptions[field_path] = field_schema["description"]
                    # Recursively extract from the referenced schema
                    extract_from_schema(ref_schema, field_path, defs_dict)
                continue

            # Extract description if present
            if "description" in field_schema:
                descriptions[field_path] = field_schema["description"]

            # Handle nested objects
            if "properties" in field_schema:
                extract_from_schema(field_schema, field_path, defs_dict)

            # Handle arrays of objects
            if "items" in field_schema:
                items_schema = field_schema["items"]
                if isinstance(items_schema, dict):
                    # Handle $ref in items
                    if "$ref" in items_schema:
                        ref_schema = resolve_ref(items_schema["$ref"], defs_dict)
                        if ref_schema:
                            extract_from_schema(ref_schema, field_path, defs_dict)
                    elif "properties" in items_schema:
                        # For arrays, we use the field path as-is (not with [])
                        extract_from_schema(items_schema, field_path, defs_dict)

    defs = schema.get("$defs", {})
    extract_from_schema(schema, prefix, defs)
    return descriptions


def apply_optimized_descriptions(
    model: type[BaseModel], optimized_descriptions: dict[str, str]
) -> dict[str, Any]:
    """Create a modified JSON schema with optimized field descriptions.
    
    This function creates a new JSON schema dictionary with updated field descriptions
    that can be used with OpenAI's structured outputs or other systems that accept
    JSON schemas.
    
    Args:
        model: The original Pydantic model class.
        optimized_descriptions: Dictionary mapping field paths to optimized descriptions.
    
    Returns:
        Modified JSON schema as a dictionary. For OpenAI, this should be wrapped in:
        {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "schema": <returned_schema>
            }
        }
    
    Example:
        ```python
        optimized = {"name": "The complete full name of the user"}
        schema = apply_optimized_descriptions(User, optimized)
        ```
    """
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})

    def resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve a $ref reference."""
        if ref.startswith("#/$defs/"):
            ref_name = ref.replace("#/$defs/", "")
            return defs.get(ref_name)
        return None

    def update_descriptions(
        schema_dict: dict[str, Any], current_prefix: str = "", defs_dict: dict[str, Any] | None = None
    ) -> None:
        """Recursively update descriptions in JSON schema."""
        if defs_dict is None:
            defs_dict = {}
        properties = schema_dict.get("properties", {})
        for field_name, field_schema in properties.items():
            field_path = (
                f"{current_prefix}.{field_name}" if current_prefix else field_name
            )

            # Handle $ref references (Pydantic v2 nested models)
            if "$ref" in field_schema:
                ref_schema = resolve_ref(field_schema["$ref"], defs_dict)
                if ref_schema:
                    # Update description from the field itself if present
                    if field_path in optimized_descriptions:
                        field_schema["description"] = optimized_descriptions[field_path]
                    # Recursively update the referenced schema (modify in place)
                    # The ref_schema is a reference to the schema in defs, so modifications persist
                    update_descriptions(ref_schema, field_path, defs_dict)
                continue

            # Update description if optimized version exists
            if field_path in optimized_descriptions:
                field_schema["description"] = optimized_descriptions[field_path]

            # Handle nested objects
            if "properties" in field_schema:
                update_descriptions(field_schema, field_path, defs_dict)

            # Handle arrays of objects
            if "items" in field_schema:
                items_schema = field_schema["items"]
                if isinstance(items_schema, dict):
                    # Handle $ref in items
                    if "$ref" in items_schema:
                        ref_schema = resolve_ref(items_schema["$ref"], defs_dict)
                        if ref_schema:
                            update_descriptions(ref_schema, field_path, defs_dict)
                    elif "properties" in items_schema:
                        update_descriptions(items_schema, field_path, defs_dict)

    # Create a deep copy to avoid modifying the original
    modified_schema = copy.deepcopy(schema)
    modified_defs = modified_schema.get("$defs", {})
    update_descriptions(modified_schema, "", modified_defs)

    return modified_schema

