"""Tests for extractor module."""

from pydantic import BaseModel, Field

from dspydantic.extractor import (
    apply_optimized_descriptions,
    create_optimized_model,
    extract_field_descriptions,
)


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


def test_create_optimized_model_simple() -> None:
    """Test creating an optimized model with updated field descriptions."""
    optimized = {
        "name": "The complete full name of the user",
        "age": "The user's age in years",
    }
    OptimizedUser = create_optimized_model(SimpleUser, optimized)

    # Verify it's a different class but with same name
    assert OptimizedUser is not SimpleUser
    assert OptimizedUser.__name__ == SimpleUser.__name__

    # Verify the optimized descriptions are in the Field definitions
    name_field = OptimizedUser.model_fields["name"]
    age_field = OptimizedUser.model_fields["age"]

    assert name_field.description == "The complete full name of the user"
    assert age_field.description == "The user's age in years"

    # Verify the model still works correctly
    user = OptimizedUser(name="John Doe", age=30)
    assert user.name == "John Doe"
    assert user.age == 30


def test_create_optimized_model_nested() -> None:
    """Test creating an optimized model with nested models."""
    optimized = {
        "name": "The complete full name",
        "address.street": "The street address",
        "address.city": "The city name",
    }
    OptimizedNestedUser = create_optimized_model(NestedUser, optimized)

    # Verify it's a different class
    assert OptimizedNestedUser is not NestedUser

    # Verify top-level field description
    name_field = OptimizedNestedUser.model_fields["name"]
    assert name_field.description == "The complete full name"

    # Verify nested model has optimized descriptions
    address_field = OptimizedNestedUser.model_fields["address"]
    address_type = address_field.annotation

    # The address_type should be an optimized Address model
    assert issubclass(address_type, BaseModel)
    assert address_type.model_fields["street"].description == "The street address"
    assert address_type.model_fields["city"].description == "The city name"

    # Verify the model still works correctly
    user = OptimizedNestedUser(
        name="John Doe",
        address={"street": "123 Main St", "city": "New York"},
    )
    assert user.name == "John Doe"
    assert user.address.street == "123 Main St"
    assert user.address.city == "New York"


def test_create_optimized_model_partial() -> None:
    """Test creating an optimized model when only some fields are optimized."""
    optimized = {"name": "The complete full name"}
    OptimizedUser = create_optimized_model(SimpleUser, optimized)

    # Verify optimized field has new description
    name_field = OptimizedUser.model_fields["name"]
    assert name_field.description == "The complete full name"

    # Verify non-optimized field keeps original description
    age_field = OptimizedUser.model_fields["age"]
    assert age_field.description == "User's age"

    # Verify the model still works correctly
    user = OptimizedUser(name="John Doe", age=30)
    assert user.name == "John Doe"
    assert user.age == 30


def test_create_optimized_model_preserves_field_attributes() -> None:
    """Test that creating an optimized model preserves other field attributes."""

    class UserWithConstraints(BaseModel):
        """User model with field constraints."""

        name: str = Field(
            description="User name",
            min_length=1,
            max_length=100,
        )
        age: int = Field(
            description="User age",
            ge=0,
            le=150,
        )

    optimized = {
        "name": "The complete full name of the user",
        "age": "The user's age in years",
    }
    OptimizedUser = create_optimized_model(UserWithConstraints, optimized)

    # Verify descriptions are updated
    name_field = OptimizedUser.model_fields["name"]
    age_field = OptimizedUser.model_fields["age"]

    assert name_field.description == "The complete full name of the user"
    assert age_field.description == "The user's age in years"

    # Verify constraints are preserved by checking JSON schema
    schema = OptimizedUser.model_json_schema()
    name_schema = schema["properties"]["name"]
    age_schema = schema["properties"]["age"]

    assert name_schema["minLength"] == 1
    assert name_schema["maxLength"] == 100
    assert age_schema["minimum"] == 0
    assert age_schema["maximum"] == 150

    # Verify validation still works
    user = OptimizedUser(name="John Doe", age=30)
    assert user.name == "John Doe"
    assert user.age == 30

    # Verify constraints are enforced (validation should fail for invalid values)
    from pydantic import ValidationError

    # Test min_length constraint
    try:
        OptimizedUser(name="", age=30)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # Test ge constraint
    try:
        OptimizedUser(name="John", age=-1)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
