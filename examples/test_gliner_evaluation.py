"""Test script to verify GLiNER2 evaluation handles entities and relations correctly."""

from dspydantic.evaluators.gliner import (
    _calculate_gliner_score,
    _normalize_expected_output,
    _normalize_gliner_result,
)

# Test case from user's extraction result
extracted_result = {
    "entities": {
        "person": ["Tim Cook"],
        "organization": ["Apple"],
        "location": ["Cupertino", "California"],
    },
    "relation_extraction": {
        "works_for": [("Tim Cook", "Apple")],
        "founded": [],
        "located_in": [("Apple", "Cupertino, California")],
        "acquired": [],
    },
}

expected_output = {
    "entities": {
        "person": ["Tim Cook"],
        "organization": ["Apple"],
        "location": ["Cupertino", "California"],
    },
    "relations": {
        "works_for": [("Tim Cook", "Apple")],
        "located_in": [("Apple", "Cupertino")],
    },
}

print("=" * 60)
print("Testing GLiNER2 Evaluation")
print("=" * 60)

print("\n1. Extracted result (raw from GLiNER2):")
print(f"   {extracted_result}")

print("\n2. Expected output:")
print(f"   {expected_output}")

# Normalize both
normalized_extracted = _normalize_gliner_result(extracted_result)
normalized_expected = _normalize_expected_output(expected_output)

print("\n3. Normalized extracted result:")
print(f"   {normalized_extracted}")

print("\n4. Normalized expected output:")
print(f"   {normalized_expected}")

# Calculate score
score = _calculate_gliner_score(normalized_expected, normalized_extracted, metric="exact")

print("\n5. Evaluation score:")
print(f"   Score: {score:.2%}")

# Detailed breakdown
print("\n6. Detailed comparison:")

# Entities comparison
extracted_entities = normalized_extracted.get("entities", {})
expected_entities = normalized_expected.get("entities", {})
print("\n   Entities:")
for entity_type in set(extracted_entities.keys()) | set(expected_entities.keys()):
    extracted_list = extracted_entities.get(entity_type, [])
    expected_list = expected_entities.get(entity_type, [])
    extracted_set = set(extracted_list)
    expected_set = set(expected_list)
    match = extracted_set == expected_set
    intersection = len(extracted_set & expected_set)
    union = len(extracted_set | expected_set)
    jaccard = intersection / union if union > 0 else 1.0
    print(f"     {entity_type}:")
    print(f"       Extracted: {extracted_list}")
    print(f"       Expected:  {expected_list}")
    print(f"       Match: {match} (Jaccard: {jaccard:.2%})")

# Relations comparison
extracted_relations = normalized_extracted.get("relations", {})
expected_relations = normalized_expected.get("relations", {})
print("\n   Relations:")
for relation_type in set(extracted_relations.keys()) | set(expected_relations.keys()):
    extracted_list = extracted_relations.get(relation_type, [])
    expected_list = expected_relations.get(relation_type, [])

    # Normalize to tuples for comparison
    def to_tuple(rel):
        if isinstance(rel, dict):
            return (rel.get("subject", ""), rel.get("object", ""))
        elif isinstance(rel, tuple) and len(rel) == 2:
            return rel
        return ("", "")

    extracted_set = {to_tuple(rel) for rel in extracted_list}
    expected_set = {to_tuple(rel) for rel in expected_list}
    match = extracted_set == expected_set
    intersection = len(extracted_set & expected_set)
    union = len(extracted_set | expected_set)
    jaccard = intersection / union if union > 0 else 1.0
    print(f"     {relation_type}:")
    print(f"       Extracted: {extracted_list}")
    print(f"       Expected:  {expected_list}")
    print(f"       Match: {match} (Jaccard: {jaccard:.2%})")
    if not match:
        print(f"       Extracted tuples: {extracted_set}")
        print(f"       Expected tuples: {expected_set}")
        print(f"       Missing: {expected_set - extracted_set}")
        print(f"       Extra: {extracted_set - expected_set}")

print("\n" + "=" * 60)
