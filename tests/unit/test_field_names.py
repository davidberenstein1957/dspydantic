"""Quick test to verify field names are used as descriptions when missing."""
from pydantic import BaseModel, Field

from dspydantic.extractor import extract_field_descriptions


class TestModel(BaseModel):
    """Test model with mixed descriptions."""
    name: str  # No description
    age: int = Field(description="User age")  # Has description
    email: str  # No description


descriptions = extract_field_descriptions(TestModel)
print("Field descriptions:", descriptions)
assert descriptions["name"] == "name", f"Expected 'name', got '{descriptions['name']}'"
assert descriptions["age"] == "User age", f"Expected 'User age', got '{descriptions['age']}'"
assert descriptions["email"] == "email", f"Expected 'email', got '{descriptions['email']}'"
print("✓ All assertions passed!")
