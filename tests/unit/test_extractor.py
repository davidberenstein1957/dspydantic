"""Tests for extractor module."""

from pydantic import BaseModel, Field

from dspydantic.extractor import apply_optimized_descriptions, extract_field_descriptions


class SimpleUser(BaseModel):
    """Simple user model for testing."""

    name: str = Field(description="User's full name")
    age: int = Field(description="User's age")


class Address(BaseModel):
    """Address model for testing."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")


class NestedUser(BaseModel):
    """User model with nested address."""

    name: str = Field(description="User's full name")
    address: Address = Field(description="User address")


def test_extract_field_descriptions_simple() -> None:
    """Test extracting field descriptions from a simple model."""
    descriptions = extract_field_descriptions(SimpleUser)
    assert "name" in descriptions
    assert "age" in descriptions
    assert descriptions["name"] == "User's full name"
    assert descriptions["age"] == "User's age"


def test_extract_field_descriptions_nested() -> None:
    """Test extracting field descriptions from a nested model."""
    descriptions = extract_field_descriptions(NestedUser)
    assert "name" in descriptions
    assert "address.street" in descriptions
    assert "address.city" in descriptions
    assert descriptions["name"] == "User's full name"
    assert descriptions["address.street"] == "Street address"
    assert descriptions["address.city"] == "City name"


def test_apply_optimized_descriptions() -> None:
    """Test applying optimized descriptions to a model."""
    optimized = {
        "name": "The complete full name of the user",
        "age": "The user's age in years",
    }
    schema = apply_optimized_descriptions(SimpleUser, optimized)
    
    assert schema["properties"]["name"]["description"] == "The complete full name of the user"
    assert schema["properties"]["age"]["description"] == "The user's age in years"


def test_apply_optimized_descriptions_nested() -> None:
    """Test applying optimized descriptions to a nested model."""
    optimized = {
        "name": "The complete full name",
        "address.street": "The street address",
        "address.city": "The city name",
    }
    schema = apply_optimized_descriptions(NestedUser, optimized)
    
    assert schema["properties"]["name"]["description"] == "The complete full name"
    # Nested models use $ref, so check $defs
    address_def = schema["$defs"]["Address"]
    assert address_def["properties"]["street"]["description"] == "The street address"
    assert address_def["properties"]["city"]["description"] == "The city name"


def test_apply_optimized_descriptions_partial() -> None:
    """Test applying optimized descriptions when only some fields are optimized."""
    optimized = {"name": "The complete full name"}
    schema = apply_optimized_descriptions(SimpleUser, optimized)
    
    assert schema["properties"]["name"]["description"] == "The complete full name"
    # Age should still have its original description
    assert schema["properties"]["age"]["description"] == "User's age"

