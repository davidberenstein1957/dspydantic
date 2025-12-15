"""GLiNER2 Named Entity Recognition (NER) example.

This example demonstrates how to optimize GLiNER2 schema descriptions for extracting
named entities from text. It optimizes entity descriptions to improve extraction accuracy.
"""

from gliner2 import GLiNER2

from dspydantic import Example, GLiNER2SchemaOptimizer


def main():
    """Run the GLiNER2 NER optimization example."""
    print("=" * 60)
    print("GLiNER2 Named Entity Recognition (NER) Example")
    print("=" * 60)
    print("\nThis example optimizes GLiNER2 entity descriptions for extracting:")
    print("- person: Names of people")
    print("- location: Places and locations")
    print("- organization: Companies and organizations")
    print("- product: Products and services\n")

    # Initialize GLiNER2 extractor
    print("Loading GLiNER2 model...")
    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("✓ Model loaded\n")

    # Define initial schema with basic descriptions
    schema = {
        "entities": {
            "person": "Names of people",
            "location": "Places",
            "organization": "Companies",
            "product": "Products",
        }
    }

    # Create examples
    examples = [
        Example(
            text="Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.",
            expected_output={
                "entities": {
                    "person": ["Tim Cook"],
                    "location": ["Cupertino"],
                    "organization": ["Apple"],
                    "product": ["iPhone 15"],
                }
            },
        ),
        Example(
            text="Microsoft's Satya Nadella introduced Azure cloud services in Seattle.",
            expected_output={
                "entities": {
                    "person": ["Satya Nadella"],
                    "location": ["Seattle"],
                    "organization": ["Microsoft"],
                    "product": ["Azure cloud services"],
                }
            },
        ),
        Example(
            text="Elon Musk founded SpaceX in Hawthorne, California.",
            expected_output={
                "entities": {
                    "person": ["Elon Musk"],
                    "location": ["Hawthorne", "California"],
                    "organization": ["SpaceX"],
                }
            },
        ),
        Example(
            text="Amazon launched AWS Lambda service from their Seattle headquarters.",
            expected_output={
                "entities": {
                    "location": ["Seattle"],
                    "organization": ["Amazon"],
                    "product": ["AWS Lambda service"],
                }
            },
        ),
        Example(
            text="Google's Sundar Pichai unveiled Pixel 8 phone in Mountain View.",
            expected_output={
                "entities": {
                    "person": ["Sundar Pichai"],
                    "location": ["Mountain View"],
                    "organization": ["Google"],
                    "product": ["Pixel 8 phone"],
                }
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
            print("  Expected entities:")
            for entity_type, entity_values in entities.items():
                print(f"    {entity_type}: {entity_values}")

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
    print("\nOptimized entity descriptions:")
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
    test_text = "OpenAI CEO Sam Altman announced GPT-5 at their San Francisco headquarters."
    print(f"\nTest text: {test_text}")

    # Build GLiNER2 schema from optimized schema dict
    gliner_schema = extractor.create_schema().entities(optimized_schema["entities"])
    extraction_result = extractor.extract(test_text, gliner_schema)

    print("\nExtraction result:")
    print(f"  {extraction_result}")


if __name__ == "__main__":
    main()
