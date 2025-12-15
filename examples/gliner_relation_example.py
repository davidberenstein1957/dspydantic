"""GLiNER2 Relation Extraction example.

This example demonstrates how to optimize GLiNER2 schema descriptions for extracting
relationships between entities. It optimizes relation descriptions to improve accuracy.
"""

from gliner2 import GLiNER2

from dspydantic import Example, GLiNER2SchemaOptimizer


def main():
    """Run the GLiNER2 relation extraction optimization example."""
    print("=" * 60)
    print("GLiNER2 Relation Extraction Example")
    print("=" * 60)
    print("\nThis example optimizes GLiNER2 relation descriptions for extracting:")
    print("- works_for: Employment relationships")
    print("- founded: Founding relationships")
    print("- located_in: Geographic relationships")
    print("- acquired: Acquisition relationships\n")

    # Initialize GLiNER2 extractor
    print("Loading GLiNER2 model...")
    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("✓ Model loaded\n")

    # Define initial schema with basic descriptions
    schema = {
        "entities": {
            "person": "Names of people",
            "organization": "Companies and organizations",
            "location": "Places",
        },
        "relations": {
            "works_for": "Employment",
            "founded": "Founding",
            "located_in": "Location",
            "acquired": "Acquisition",
        },
    }

    # Create examples
    examples = [
        Example(
            text="John works for Apple Inc. and lives in San Francisco. Apple Inc. is located in Cupertino.",
            expected_output={
                "entities": {
                    "person": ["John"],
                    "organization": ["Apple Inc."],
                    "location": ["San Francisco", "Cupertino"],
                },
                "relations": {
                    "works_for": [("John", "Apple Inc.")],
                    "located_in": [("Apple Inc.", "Cupertino")],
                },
            },
        ),
        Example(
            text="Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne, California.",
            expected_output={
                "entities": {
                    "person": ["Elon Musk"],
                    "organization": ["SpaceX"],
                    "location": ["Hawthorne", "California"],
                },
                "relations": {
                    "founded": [("Elon Musk", "SpaceX")],
                    "located_in": [("SpaceX", "Hawthorne, California")],
                },
            },
        ),
        Example(
            text="Microsoft acquired GitHub in 2018. GitHub is located in San Francisco.",
            expected_output={
                "entities": {
                    "organization": ["Microsoft", "GitHub"],
                    "location": ["San Francisco"],
                },
                "relations": {
                    "acquired": [("Microsoft", "GitHub")],
                    "located_in": [("GitHub", "San Francisco")],
                },
            },
        ),
        Example(
            text="Satya Nadella works for Microsoft. Microsoft is located in Redmond, Washington.",
            expected_output={
                "entities": {
                    "person": ["Satya Nadella"],
                    "organization": ["Microsoft"],
                    "location": ["Redmond", "Washington"],
                },
                "relations": {
                    "works_for": [("Satya Nadella", "Microsoft")],
                    "located_in": [("Microsoft", "Redmond, Washington")],
                },
            },
        ),
        Example(
            text="Mark Zuckerberg founded Facebook in 2004. Facebook acquired Instagram in 2012.",
            expected_output={
                "entities": {
                    "person": ["Mark Zuckerberg"],
                    "organization": ["Facebook", "Instagram"],
                },
                "relations": {
                    "founded": [("Mark Zuckerberg", "Facebook")],
                    "acquired": [("Facebook", "Instagram")],
                },
            },
        ),
    ]

    print(f"Created {len(examples)} examples\n")
    print("Sample examples:")
    for i, example in enumerate(examples[:2], 1):
        print(f"\nExample {i}:")
        text_preview = example.input_data.get("text", "")
        print(f"  Text: {text_preview}")
        expected = example.expected_output
        if isinstance(expected, dict):
            entities = expected.get("entities", {})
            relations = expected.get("relations", {})
            print("  Expected entities:")
            for entity_type, entity_values in entities.items():
                print(f"    {entity_type}: {entity_values}")
            print("  Expected relations:")
            for rel_type, rel_values in relations.items():
                print(f"    {rel_type}: {rel_values}")

    # Create optimizer
    print("\n" + "=" * 60)
    print("Creating GLiNER2 Schema Optimizer...")
    print("=" * 60)
    optimizer = GLiNER2SchemaOptimizer(
        extractor=extractor,
        schema=schema,
        examples=examples,
        model_id="gpt-4o-mini",
        verbose=True,
    )

    # Optimize
    print("\n" + "=" * 60)
    print("Starting optimization...")
    print("=" * 60)
    result = optimizer.optimize()

    # Print results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Baseline score: {result.baseline_score:.2%}")
    print(f"Optimized score: {result.optimized_score:.2%}")
    print(f"Improvement: {result.metrics['improvement']:+.2%}")
    print("\nOptimized descriptions:")
    for element_path, description in result.optimized_descriptions.items():
        print(f"\n  {element_path}:")
        print(f"    {description}")

    # Apply optimized schema
    print("\n" + "=" * 60)
    print("Applying optimized schema...")
    print("=" * 60)
    optimized_schema = optimizer.apply_optimized_schema()
    print("\nOptimized schema:")
    print(f"  {optimized_schema}")

    # Test extraction with optimized schema
    print("\n" + "=" * 60)
    print("Testing extraction with optimized schema...")
    print("=" * 60)
    test_text = "Tim Cook works for Apple. Apple is located in Cupertino, California."
    print(f"\nTest text: {test_text}")

    # Build GLiNER2 schema from optimized schema dict
    gliner_schema = extractor.create_schema()
    if "entities" in optimized_schema:
        gliner_schema = gliner_schema.entities(optimized_schema["entities"])
    if "relations" in optimized_schema:
        gliner_schema = gliner_schema.relations(optimized_schema["relations"])

    extraction_result = extractor.extract(test_text, gliner_schema)

    print("\nExtraction result:")
    print(f"  {extraction_result}")


if __name__ == "__main__":
    main()
