import json
from typing import Dict, Any, Optional, Literal, List, Union, get_origin, get_args
from pydantic import BaseModel, create_model, ValidationError, Field
from pydantic.json_schema import model_json_schema


def _create_literal_type(values: List[Any]) -> type:
    """Create a Literal type from a list of values."""
    if len(values) == 1:
        return Literal[values[0]]
    # For multiple values, construct Literal[val1, val2, ...]
    # In Python, Literal[val1, val2] is equivalent to Literal.__class_getitem__((val1, val2))
    # where the tuple contains the individual values as separate arguments
    try:
        # Try the Python 3.9+ way
        return Literal.__class_getitem__(tuple(values))
    except (TypeError, AttributeError):
        # Fallback: use Union of individual Literals (works but less ideal)
        literal_types = [Literal[val] for val in values]
        if len(literal_types) == 1:
            return literal_types[0]
        return Union[tuple(literal_types)]


def create_pydantic_model_from_schema(schema: Dict[str, Any]) -> type[BaseModel]:
    """Dynamically create a Pydantic model from a JSON schema with support for constraints."""
    fields = {}
    
    if "properties" not in schema:
        raise ValueError("Schema must have 'properties' field")
    
    for field_name, field_def in schema["properties"].items():
        field_type, field_info = _build_field_from_schema(field_def, field_name, schema.get("required", []))
        fields[field_name] = (field_type, field_info)
    
    return create_model("DynamicModel", **fields)


def _build_field_from_schema(field_def: Dict[str, Any], field_name: str, required_fields: List[str]) -> tuple[Any, Any]:
    """Build a Pydantic field from a JSON schema field definition with constraints."""
    field_type_str = field_def.get("type")
    is_required = field_name in required_fields
    
    # Handle enum for strings
    if field_type_str == "string" and "enum" in field_def and field_def["enum"]:
        enum_values = field_def["enum"]
        field_type = _create_literal_type(enum_values)
        default = ... if is_required else None
        return field_type, default
    
    # Handle arrays
    if field_type_str == "array":
        items_def = field_def.get("items", {})
        if items_def:
            item_type, item_default = _build_field_from_schema(items_def, "item", [])
            # For arrays, we use List[item_type]
            field_type = List[item_type]
        else:
            field_type = List[Any]
        default = ... if is_required else None
        return field_type, default
    
    # Handle objects (nested models)
    if field_type_str == "object":
        if "properties" in field_def and field_def["properties"]:
            # Create a nested model
            nested_fields = {}
            nested_required = field_def.get("required", [])
            for nested_name, nested_def in field_def["properties"].items():
                nested_type, nested_info = _build_field_from_schema(nested_def, nested_name, nested_required)
                nested_fields[nested_name] = (nested_type, nested_info)
            nested_model = create_model(f"DynamicModel_{field_name}", **nested_fields)
            default = ... if is_required else None
            return nested_model, default
        else:
            # Empty object or dict
            field_type = Dict[str, Any]
            default = ... if is_required else None
            return field_type, default
    
    # Handle primitive types with constraints
    field_type = _get_python_type(field_type_str)
    field_kwargs = {}
    
    # Add min/max constraints for integers and numbers
    if field_type_str in ("integer", "number"):
        if "minimum" in field_def:
            field_kwargs["ge"] = field_def["minimum"]
        if "maximum" in field_def:
            field_kwargs["le"] = field_def["maximum"]
    
    # Add description if present
    if "description" in field_def:
        field_kwargs["description"] = field_def["description"]
    
    # Create Field with constraints if any, otherwise use default
    if field_kwargs:
        if is_required:
            default = Field(..., **field_kwargs)
        else:
            default = Field(default=None, **field_kwargs)
    else:
        default = ... if is_required else None
    
    return field_type, default


def _get_python_type(json_type: str) -> type:
    """Convert JSON schema type to Python type."""
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_mapping.get(json_type, str)


def validate_data_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, Optional[str], Optional[Dict]]:
    """
    Validate data against a Pydantic schema.
    Returns: (is_valid, error_message, validated_data)
    """
    try:
        model = create_pydantic_model_from_schema(schema)
        validated = model(**data)
        return True, None, validated.model_dump()
    except ValidationError as e:
        return False, str(e), None
    except Exception as e:
        return False, f"Schema error: {str(e)}", None


def schema_to_json_schema(pydantic_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Pydantic schema dict to JSON schema format."""
    try:
        model = create_pydantic_model_from_schema(pydantic_schema)
        result = model_json_schema(model)
        # model_json_schema returns a dict in newer Pydantic versions, or (dict, dict) in older versions
        if isinstance(result, tuple):
            json_schema, _ = result
        else:
            json_schema = result
        return json_schema
    except Exception as e:
        raise ValueError(f"Failed to convert schema: {str(e)}")
