"""GLiNER2 Classification example.

This example demonstrates how to optimize GLiNER2 schema descriptions for text
classification tasks. It optimizes classification label descriptions to improve accuracy.
"""

from gliner2 import GLiNER2

from dspydantic import Example, GLiNER2SchemaOptimizer


def main():
    """Run the GLiNER2 classification optimization example."""
    print("=" * 60)
    print("GLiNER2 Classification Example")
    print("=" * 60)
    print("\nThis example optimizes GLiNER2 classification descriptions for:")
    print("- sentiment: positive, negative, or neutral")
    print("- category: technology, business, finance, or healthcare\n")

    # Initialize GLiNER2 extractor
    print("Loading GLiNER2 model...")
    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("✓ Model loaded\n")

    # Define initial schema with basic descriptions
    schema = {
        "classifications": {
            "sentiment": {
                "labels": ["positive", "negative", "neutral"],
                "multi_label": False,
            },
            "category": {
                "labels": {
                    "technology": "Tech topics",
                    "business": "Business topics",
                    "finance": "Finance topics",
                    "healthcare": "Healthcare topics",
                },
                "multi_label": False,
            },
        }
    }

    # Create examples
    examples = [
        Example(
            text="This laptop has amazing performance but terrible battery life!",
            expected_output={
                "sentiment": "negative",
                "category": "technology",
            },
        ),
        Example(
            text="Great camera quality, decent performance, but poor battery life.",
            expected_output={
                "sentiment": "negative",
                "category": "technology",
            },
        ),
        Example(
            text="The company's quarterly earnings exceeded expectations significantly.",
            expected_output={
                "sentiment": "positive",
                "category": "business",
            },
        ),
        Example(
            text="Stock prices dropped sharply following the market announcement.",
            expected_output={
                "sentiment": "negative",
                "category": "finance",
            },
        ),
        Example(
            text="New breakthrough treatment shows promising results in clinical trials.",
            expected_output={
                "sentiment": "positive",
                "category": "healthcare",
            },
        ),
        Example(
            text="The product launch was successful and well-received by customers.",
            expected_output={
                "sentiment": "positive",
                "category": "business",
            },
        ),
        Example(
            text="Market volatility continues to concern investors this quarter.",
            expected_output={
                "sentiment": "negative",
                "category": "finance",
            },
        ),
        Example(
            text="The new smartphone features impressive technology and innovation.",
            expected_output={
                "sentiment": "positive",
                "category": "technology",
            },
        ),
    ]

    print(f"Created {len(examples)} examples\n")
    print("Sample examples:")
    for i, example in enumerate(examples[:3], 1):
        print(f"\nExample {i}:")
        text_preview = example.input_data.get("text", "")
        print(f"  Text: {text_preview}")
        expected = example.expected_output
        if isinstance(expected, dict):
            sentiment = expected.get("sentiment", "")
            category = expected.get("category", "")
            print(f"  Expected sentiment: {sentiment}")
            print(f"  Expected category: {category}")

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
    print("\nOptimized classification descriptions:")
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
    test_text = "The new AI model demonstrates exceptional capabilities in natural language processing."
    print(f"\nTest text: {test_text}")

    # Build GLiNER2 schema from optimized schema dict
    gliner_schema = extractor.create_schema()
    for label_name, label_config in optimized_schema["classifications"].items():
        labels = label_config.get("labels", [])
        multi_label = label_config.get("multi_label", False)
        cls_threshold = label_config.get("cls_threshold", 0.5)
        gliner_schema = gliner_schema.classification(
            label_name, labels, multi_label=multi_label, cls_threshold=cls_threshold
        )

    extraction_result = extractor.extract(test_text, gliner_schema)

    print("\nExtraction result:")
    print(f"  {extraction_result}")


if __name__ == "__main__":
    main()
