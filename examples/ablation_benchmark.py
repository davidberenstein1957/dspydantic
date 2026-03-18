"""Ablation study comparing different optimization strategies.

This script benchmarks four optimization approaches:
1. Single-pass (default): One DSPy compile for all fields with reduced demos
2. Sequential: One compile per field, deepest-first (old v0.1.3 behavior)
3. Sequential + Parallel: One compile per field, all run simultaneously
4. Sequential + Max Val Examples: Sequential with validation set capped at 5

Requires OPENAI_API_KEY environment variable to be set.
"""

import os
import time
from typing import Any

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


class Product(BaseModel):
    """Product information extraction model."""

    name: str = Field(description="Product name")
    price: float = Field(description="Product price in USD")
    category: str = Field(description="Product category")


def load_examples() -> list[Example]:
    """Load sample product extraction examples."""
    return [
        Example(
            text="The Apple iPhone 15 Pro Max costs $1199.99 and is in the electronics category.",
            expected_output={"name": "Apple iPhone 15 Pro Max", "price": 1199.99, "category": "electronics"},
        ),
        Example(
            text="Nike Air Max sneakers are priced at $120.00 in the footwear section.",
            expected_output={"name": "Nike Air Max", "price": 120.00, "category": "footwear"},
        ),
        Example(
            text="A Samsung 65-inch 4K TV retails for $799.99 in home electronics.",
            expected_output={"name": "Samsung 65-inch 4K TV", "price": 799.99, "category": "home electronics"},
        ),
        Example(
            text="Dyson V15 Detect cordless vacuum costs $749.99 and is in the appliances department.",
            expected_output={"name": "Dyson V15 Detect", "price": 749.99, "category": "appliances"},
        ),
        Example(
            text="Sony WH-1000XM5 headphones are $399.99 in the audio section.",
            expected_output={"name": "Sony WH-1000XM5", "price": 399.99, "category": "audio"},
        ),
        Example(
            text="LG OLED 55-inch TV is priced at $1699.99 in home theater.",
            expected_output={"name": "LG OLED 55-inch TV", "price": 1699.99, "category": "home theater"},
        ),
        Example(
            text="Dell XPS 15 laptop costs $1999.99 and is under computers.",
            expected_output={"name": "Dell XPS 15", "price": 1999.99, "category": "computers"},
        ),
        Example(
            text="The DJI Mini 3 Pro drone is $399.00 in the drones & accessories category.",
            expected_output={"name": "DJI Mini 3 Pro", "price": 399.00, "category": "drones & accessories"},
        ),
        Example(
            text="Bose QuietComfort 45 earbuds retail for $299.99 in audio.",
            expected_output={"name": "Bose QuietComfort 45", "price": 299.99, "category": "audio"},
        ),
        Example(
            text="Apple AirPods Pro are $249.99 in the audio and accessories department.",
            expected_output={"name": "Apple AirPods Pro", "price": 249.99, "category": "audio and accessories"},
        ),
        Example(
            text="Google Pixel 8 Pro smartphone is priced at $999.99 in mobile phones.",
            expected_output={"name": "Google Pixel 8 Pro", "price": 999.99, "category": "mobile phones"},
        ),
        Example(
            text="The iPad Pro 12.9-inch (2024) costs $1099.00 and is in tablets.",
            expected_output={"name": "iPad Pro 12.9-inch", "price": 1099.00, "category": "tablets"},
        ),
    ]


def run_optimization(
    name: str,
    examples: list[Example],
    **optimizer_kwargs: Any,
) -> dict[str, Any]:
    """Run a single optimization configuration and record metrics.

    Args:
        name: Configuration name
        examples: List of examples
        **optimizer_kwargs: Arguments to pass to PydanticOptimizer

    Returns:
        Dictionary with timing and quality metrics
    """
    import dspy

    # Configure DSPy
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key, cache=False)
    dspy.configure(lm=lm)

    print(f"\n{'='*60}")
    print(f"Configuration: {name}")
    print(f"{'='*60}")

    start = time.perf_counter()
    try:
        optimizer = PydanticOptimizer(
            model=Product,
            examples=examples,
            evaluate_fn="exact",
            num_threads=4,
            verbose=True,
            **optimizer_kwargs,
        )
        result = optimizer.optimize()
        elapsed = time.perf_counter() - start

        # Count API calls if available
        api_calls = 0
        if hasattr(lm, "history") and lm.history:
            api_calls = len(lm.history)

        return {
            "name": name,
            "elapsed_seconds": elapsed,
            "baseline_score": result.baseline_score,
            "optimized_score": result.optimized_score,
            "improvement": result.optimized_score - result.baseline_score,
            "api_calls": api_calls,
            "success": True,
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "name": name,
            "elapsed_seconds": elapsed,
            "baseline_score": None,
            "optimized_score": None,
            "improvement": None,
            "api_calls": None,
            "success": False,
            "error": str(e),
        }


def main():
    """Run ablation study comparing optimization strategies."""
    print("Loading examples...")
    examples = load_examples()
    print(f"Loaded {len(examples)} examples")

    configs = [
        # ("Single-pass (default)", {"sequential": False}),
        # ("Sequential", {"sequential": True, "parallel_fields": False}),
        ("Sequential + Parallel", {"sequential": False, "parallel_fields": True}),
        # ("Sequential + Max Val=5", {"sequential": True, "parallel_fields": False, "max_val_examples": 5}),
    ]

    results = []
    for name, kwargs in configs:
        result = run_optimization(name, examples, **kwargs)
        results.append(result)

    # Print summary table
    print(f"\n\n{'='*100}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*100}\n")

    print(f"{'Config':<30} {'Time (s)':<12} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12} {'API Calls':<12} {'Status':<12}")
    print("-" * 100)

    for result in results:
        if result["success"]:
            print(
                f"{result['name']:<30} "
                f"{result['elapsed_seconds']:<12.2f} "
                f"{result['baseline_score']:<12.2%} "
                f"{result['optimized_score']:<12.2%} "
                f"{result['improvement']:<12.2%} "
                f"{result['api_calls']:<12} "
                f"{'✓':<12}"
            )
        else:
            print(
                f"{result['name']:<30} "
                f"{result['elapsed_seconds']:<12.2f} "
                f"{'N/A':<12} "
                f"{'N/A':<12} "
                f"{'N/A':<12} "
                f"{'N/A':<12} "
                f"{'Error':<12}"
            )
            print(f"  Error: {result['error']}")

    print()

    # Analysis
    successful = [r for r in results if r["success"]]
    if successful:
        fastest = min(successful, key=lambda r: r["elapsed_seconds"])
        best_quality = max(successful, key=lambda r: r["optimized_score"])

        print("\nINSIGHTS:")
        print(f"  Fastest: {fastest['name']} ({fastest['elapsed_seconds']:.2f}s)")
        print(f"  Best quality: {best_quality['name']} (score: {best_quality['optimized_score']:.2%})")

        # Speedup analysis
        baseline_time = results[0]["elapsed_seconds"]
        for result in results[1:]:
            if result["success"]:
                speedup = baseline_time / result["elapsed_seconds"]
                print(f"  {result['name']}: {speedup:.2f}× vs single-pass")


if __name__ == "__main__":
    main()
