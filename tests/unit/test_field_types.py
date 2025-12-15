"""Test field type extraction."""

from pydantic import BaseModel

from dspydantic.extractors import extract_field_types


class Address(BaseModel):
    street: str
    city: str


class User(BaseModel):
    name: str
    age: int
    tags: list[str]
    address: Address | None


types = extract_field_types(User)
print("Field types:", types)

# Verify expected types
assert types["name"] == "str"
assert types["age"] == "int"
assert "List" in types["tags"] or "list" in types["tags"].lower()
assert types["address.street"] == "str"
assert types["address.city"] == "str"
print("\n✓ All field types extracted correctly!")
