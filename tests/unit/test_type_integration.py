"""Test that field types are passed to optimization."""
from pydantic import BaseModel

from dspydantic import Example, PydanticOptimizer


class TestModel(BaseModel):
    """Test model without field descriptions."""
    name: str  # No description
    age: int  # No description


examples = [
    Example(
        text="John Doe, 30 years old",
        expected_output={"name": "John Doe", "age": 30}
    )
]

optimizer = PydanticOptimizer(
    model=TestModel,
    examples=examples,
    instruction_prompt="Extract name and age",
    verbose=False,
)

print("Field descriptions:", optimizer.field_descriptions)
print("Field types:", optimizer.field_types)

# Verify field types are extracted
assert "name" in optimizer.field_types
assert "age" in optimizer.field_types
assert optimizer.field_types["name"] == "str"
assert optimizer.field_types["age"] == "int"
print("\n✓ Field types correctly extracted and stored!")
