"""MNIST image classification example using ylecun/mnist dataset.

This example demonstrates how to optimize a Pydantic model for digit classification
on the MNIST handwritten digit dataset from HuggingFace.
"""

import base64
import io
import random
from typing import Literal

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


class DigitClassification(BaseModel):
    """Digit classification model for MNIST handwritten digits."""

    digit: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = Field(
        description="The digit shown in the image, a number from 0 to 9"
    )


def pil_image_to_base64(image) -> str:
    """Convert a PIL Image to base64-encoded string.

    Args:
        image: PIL Image object.

    Returns:
        Base64-encoded image string.
    """
    buffer = io.BytesIO()
    # Convert to RGB if necessary
    if image.mode != "RGB":
        rgb_image = image.convert("RGB")
    else:
        rgb_image = image
    rgb_image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return base64_str


def load_mnist_examples(num_examples: int = 10, split: str = "train") -> list[Example]:
    """Load examples from the MNIST dataset.

    Ensures at least one example for each digit (0-9).

    Args:
        num_examples: Number of examples to load (default: 10).
        split: Dataset split to use, either "train" or "test" (default: "train").

    Returns:
        List of Example objects with images and expected digit labels.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library is required. Install it with: uv pip install datasets"
        )

    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image processing. Install it with: uv pip install pillow"
        )

    # Load the MNIST dataset
    dataset = load_dataset("ylecun/mnist", split=split)
    dataset_size = len(dataset)

    # Find indices for each digit (0-9)
    digit_to_indices: dict[int, list[int]] = {digit: [] for digit in range(10)}

    for idx in range(dataset_size):
        label = dataset[idx]["label"]
        if label in digit_to_indices:
            digit_to_indices[label].append(idx)

    # Ensure at least one example for each digit
    selected_indices: set[int] = set()
    for digit in range(10):
        if digit_to_indices[digit]:
            selected_indices.add(random.choice(digit_to_indices[digit]))

    # Fill remaining slots randomly
    remaining_needed = num_examples - len(selected_indices)
    if remaining_needed > 0:
        available_indices = set(range(dataset_size)) - selected_indices
        if available_indices:
            additional_indices = random.sample(
                list(available_indices), min(remaining_needed, len(available_indices))
            )
            selected_indices.update(additional_indices)

    # Convert to list and limit to num_examples
    selected_indices_list = list(selected_indices)[:num_examples]
    random.shuffle(selected_indices_list)

    # Build examples
    examples = []
    for idx in selected_indices_list:
        item = dataset[idx]
        # Get the image (PIL Image object) and label
        image = item["image"]
        label = item["label"]

        # Convert PIL Image to base64
        image_base64 = pil_image_to_base64(image)

        example = Example(
            image_base64=image_base64,
            expected_output={"digit": label},
        )
        examples.append(example)

    return examples


def main():
    """Run the MNIST digit classification optimization example."""
    print("Loading MNIST dataset examples...")
    examples = load_mnist_examples(num_examples=10, split="train")

    print(f"Loaded {len(examples)} examples")
    print("\nSample examples:")
    for i, example in enumerate(examples[:3], 1):
        print(f"\nExample {i}:")
        print(f"  Image size: {len(example.input_data['images'][0])} base64 chars")
        print(f"  Expected digit: {example.expected_output['digit']}")

    # Create optimizer with system and instruction prompts
    optimizer = PydanticOptimizer(
        model=DigitClassification,
        examples=examples,
        model_id="gpt-4o-mini",  # Will use OPENAI_API_KEY from environment
        verbose=True,
        optimizer_type="miprov2zeroshot",
        system_prompt=(
            "You are an expert image classification assistant specializing in handwritten "
            "digit recognition. You have extensive experience analyzing MNIST-style "
            "handwritten digit images and can accurately identify digits from 0 to 9 even "
            "when the handwriting is unclear or stylized."
        ),
        instruction_prompt=(
            "Analyze the provided handwritten digit image and identify the digit shown. "
            "The digit will be a single number from 0 to 9. Look carefully at the shape, "
            "strokes, and overall form of the handwritten digit to make an accurate "
            "classification. Return the digit value as specified in the JSON schema."
        ),
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
    print("\nOptimized system prompt:")
    print(f"  {result.optimized_system_prompt}")
    print("\nOptimized instruction prompt:")
    print(f"  {result.optimized_instruction_prompt}")
    print("\nOptimized descriptions:")
    for field_path, description in result.optimized_descriptions.items():
        print(f"  {field_path}: {description}")


if __name__ == "__main__":
    main()

