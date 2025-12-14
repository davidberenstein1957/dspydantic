"""Test that field descriptions are initialized from field names."""

from pydantic import BaseModel

from dspydantic import Example, PydanticOptimizer


class TestModel(BaseModel):
    """Test model without field descriptions."""

    name: str  # No description
    age: int  # No description


examples = [Example(text="John Doe, 30 years old", expected_output={"name": "John Doe", "age": 30})]

optimizer = PydanticOptimizer(
    model=TestModel,
    examples=examples,
    instruction_prompt="Extract name and age",
    verbose=False,
)

print("Field descriptions initialized during __init__:")
for field_path, description in optimizer.field_descriptions.items():
    print(f"  {field_path}: {description}")

# Verify they are set to field names
assert optimizer.field_descriptions["name"] == "name"
assert optimizer.field_descriptions["age"] == "age"
print("\n✓ Field descriptions correctly initialized from field names!")
