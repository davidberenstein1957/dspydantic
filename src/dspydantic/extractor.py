"""Utilities for extracting and applying field descriptions from Pydantic models."""

import copy
from typing import Any, get_args, get_origin

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo


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
        schema_dict: dict[str, Any],
        current_prefix: str = "",
        defs_dict: dict[str, Any] | None = None,
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

    def resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve a $ref reference."""
        if ref.startswith("#/$defs/"):
            ref_name = ref.replace("#/$defs/", "")
            return defs.get(ref_name)
        return None

    def update_descriptions(
        schema_dict: dict[str, Any],
        current_prefix: str = "",
        defs_dict: dict[str, Any] | None = None,
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


def create_optimized_model(
    model: type[BaseModel], optimized_descriptions: dict[str, str]
) -> type[BaseModel]:
    """Create a new Pydantic model class with optimized field descriptions applied.

    This function creates a new model class with updated Field descriptions directly
    applied to the model fields. This allows you to use the optimized model directly
    in your code, with all the optimized descriptions embedded in the Field definitions.

    Args:
        model: The original Pydantic model class.
        optimized_descriptions: Dictionary mapping field paths to optimized descriptions.
            Field paths use dot notation for nested fields (e.g., "address.street").

    Returns:
        A new Pydantic model class with optimized descriptions applied.

    Example:
        ```python
        from pydantic import BaseModel, Field
        from dspydantic import create_optimized_model

        class User(BaseModel):
            name: str = Field(description="User name")
            age: int = Field(description="User age")

        optimized = {
            "name": "The complete full name of the user",
            "age": "The user's age in years"
        }

        OptimizedUser = create_optimized_model(User, optimized)
        # OptimizedUser now has the optimized descriptions in its Field definitions
        ```
    """
    # Cache for nested models to avoid recreating them
    nested_model_cache: dict[str, type[BaseModel]] = {}
    # Cache for original model JSON schemas to extract constraints
    original_schema_cache: dict[type[BaseModel], dict[str, Any]] = {}

    def get_original_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
        """Get and cache the JSON schema for a model."""
        if model_cls not in original_schema_cache:
            original_schema_cache[model_cls] = model_cls.model_json_schema()
        return original_schema_cache[model_cls]

    def get_field_path_prefix(prefix: str, field_name: str) -> str:
        """Get the full field path."""
        return f"{prefix}.{field_name}" if prefix else field_name

    def extract_constraints_from_schema(schema: dict[str, Any], field_name: str) -> dict[str, Any]:
        """Extract Field constraint kwargs from JSON schema."""
        constraints: dict[str, Any] = {}
        field_schema = schema.get("properties", {}).get(field_name, {})

        # Map JSON schema constraint names to Field kwargs
        if "minLength" in field_schema:
            constraints["min_length"] = field_schema["minLength"]
        if "maxLength" in field_schema:
            constraints["max_length"] = field_schema["maxLength"]
        if "minimum" in field_schema:
            constraints["ge"] = field_schema["minimum"]
        if "maximum" in field_schema:
            constraints["le"] = field_schema["maximum"]
        if "pattern" in field_schema:
            constraints["pattern"] = field_schema["pattern"]
        if "examples" in field_schema:
            constraints["examples"] = field_schema["examples"]

        return constraints

    def create_optimized_field(
        field_name: str,
        field_info: FieldInfo,
        field_type: Any,
        current_prefix: str = "",
    ) -> tuple[Any, FieldInfo]:
        """Create an optimized field with updated description if available.

        Args:
            field_name: Name of the field.
            field_info: Original FieldInfo object.
            field_type: Type annotation for the field.
            current_prefix: Current prefix for nested fields.

        Returns:
            Tuple of (field_type, updated_field_info).
        """
        field_path = get_field_path_prefix(current_prefix, field_name)

        # Get the original description or use optimized one
        original_description = field_info.description
        optimized_description = optimized_descriptions.get(field_path, original_description)

        # Create new FieldInfo with updated description
        # Preserve all other field attributes
        field_kwargs: dict[str, Any] = {}

        # Copy existing Field attributes
        if field_info.default is not ...:
            field_kwargs["default"] = field_info.default
        if field_info.default_factory is not None:
            field_kwargs["default_factory"] = field_info.default_factory

        # Extract constraints from the original model's JSON schema
        # We need to get the model that contains this field
        # For top-level fields, use the original model; for nested, we'll handle it differently
        if current_prefix == "":
            # Top-level field - use the original model
            model_schema = get_original_schema(model)
            constraints = extract_constraints_from_schema(model_schema, field_name)
            field_kwargs.update(constraints)

        # Copy constraints and validators
        if hasattr(field_info, "json_schema_extra"):
            field_kwargs["json_schema_extra"] = field_info.json_schema_extra

        # Handle nested models
        origin = get_origin(field_type)

        # Check if it's a nested BaseModel
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # Create optimized nested model
            optimized_nested = create_optimized_nested_model(
                field_type, field_path, nested_model_cache
            )
            field_type = optimized_nested
        elif origin is not None:
            # Handle generic types like List, Optional, etc.
            args = get_args(field_type)
            if args:
                # Check if any args are BaseModel subclasses
                new_args = []
                for arg in args:
                    if isinstance(arg, type) and issubclass(arg, BaseModel):
                        optimized_nested = create_optimized_nested_model(
                            arg, field_path, nested_model_cache
                        )
                        new_args.append(optimized_nested)
                    else:
                        new_args.append(arg)

                # Reconstruct the type with optimized nested models
                if new_args != args:
                    field_type = origin[tuple(new_args)]

        # Create Field with updated description
        if optimized_description:
            field_kwargs["description"] = optimized_description

        updated_field = Field(**field_kwargs) if field_kwargs else field_info

        return field_type, updated_field

    def create_optimized_nested_model(
        nested_model: type[BaseModel],
        parent_prefix: str,
        cache: dict[str, type[BaseModel]],
    ) -> type[BaseModel]:
        """Create an optimized version of a nested model.

        Args:
            nested_model: The nested BaseModel class.
            parent_prefix: Prefix for field paths (e.g., "address").
            cache: Cache to avoid recreating the same nested model.

        Returns:
            Optimized nested model class.
        """
        # Use cache key based on model name and prefix
        cache_key = f"{parent_prefix}:{nested_model.__name__}"
        if cache_key in cache:
            return cache[cache_key]

        # Get model fields and schema for constraint extraction
        model_fields = nested_model.model_fields
        nested_schema = get_original_schema(nested_model)

        # Build new fields dict
        new_fields: dict[str, tuple[Any, FieldInfo]] = {}

        for field_name, field_info in model_fields.items():
            field_type = nested_model.__annotations__.get(field_name)
            if field_type is None:
                continue

            # Extract constraints for nested model fields
            field_path = get_field_path_prefix(parent_prefix, field_name)
            constraints = extract_constraints_from_schema(nested_schema, field_name)

            # Get optimized description
            original_description = field_info.description
            optimized_description = optimized_descriptions.get(field_path, original_description)

            # Build field kwargs
            field_kwargs: dict[str, Any] = {}
            if field_info.default is not ...:
                field_kwargs["default"] = field_info.default
            if field_info.default_factory is not None:
                field_kwargs["default_factory"] = field_info.default_factory
            field_kwargs.update(constraints)
            if optimized_description:
                field_kwargs["description"] = optimized_description

            # Handle nested models recursively
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                optimized_nested = create_optimized_nested_model(
                    field_type, field_path, nested_model_cache
                )
                field_type = optimized_nested

            optimized_field = Field(**field_kwargs) if field_kwargs else field_info
            new_fields[field_name] = (field_type, optimized_field)

        # Create new model class
        # Use the first base class if available, otherwise BaseModel
        base = (
            nested_model.__bases__[0]
            if nested_model.__bases__ and nested_model.__bases__[0] != BaseModel
            else BaseModel
        )
        optimized_nested = create_model(  # type: ignore[call-overload]
            nested_model.__name__,
            __base__=base,
            __module__=nested_model.__module__,
            __doc__=nested_model.__doc__,
            **new_fields,
        )

        cache[cache_key] = optimized_nested
        return optimized_nested

    # Get model fields
    model_fields = model.model_fields

    # Build new fields dict with optimized descriptions
    new_fields: dict[str, tuple[Any, FieldInfo]] = {}

    for field_name, field_info in model_fields.items():
        field_type = model.__annotations__.get(field_name)
        if field_type is None:
            continue

        optimized_type, optimized_field = create_optimized_field(
            field_name, field_info, field_type, ""
        )
        new_fields[field_name] = (optimized_type, optimized_field)

    # Create new model class
    # Use the first base class if available, otherwise BaseModel
    base = model.__bases__[0] if model.__bases__ and model.__bases__[0] != BaseModel else BaseModel
    optimized_model = create_model(  # type: ignore[call-overload]
        model.__name__,
        __base__=base,
        __module__=model.__module__,
        __doc__=model.__doc__,
        **new_fields,
    )

    return optimized_model
