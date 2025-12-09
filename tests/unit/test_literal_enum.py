"""Test Literal and Enum type extraction."""
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from dspydantic.extractor import extract_field_types


class Status(str, Enum):
    """Status enum."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class TestModel(BaseModel):
    """Test model with Literal and Enum."""
    digit: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    status: Status
    name: str
    age: int


types = extract_field_types(TestModel)
print("Field types:", types)

# Verify Literal type
assert "digit" in types
assert "Literal" in types["digit"]
print(f"\n✓ Literal type extracted: {types['digit']}")

# Verify Enum type
assert "status" in types
print(f"✓ Enum type extracted: {types['status']}")

# Verify simple types still work
assert types["name"] == "str"
assert types["age"] == "int"
print("✓ Simple types still work correctly")
