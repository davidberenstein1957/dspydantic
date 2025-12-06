"""Human-in-the-Loop (HITL) evaluation example.

This example demonstrates how to use the human-in-the-loop evaluation functions
('exact-hitl' and 'levenshtein-hitl') to review and edit extracted outputs
during optimization.

When using HITL evaluation:
- A GUI popup will appear for each evaluation
- You can review the input (text/images) and proposed output
- You can edit the output JSON if needed
- The score reflects whether changes were made:
  - exact-hitl: 0.0 if edited, 1.0 if not edited
  - levenshtein-hitl: Levenshtein similarity if edited, 1.0 if not edited

This example demonstrates HITL with both text and image inputs (MNIST digits).
"""

import base64
import io
import random
from typing import Literal

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


class UserProfile(BaseModel):
    """User profile extraction model."""

    name: str = Field(description="Full name of the user")
    email: str = Field(description="Email address")
    age: int = Field(description="Age in years")
    city: str = Field(description="City of residence")


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


def load_mnist_examples(num_examples: int = 5, split: str = "train") -> list[Example]:
    """Load examples from the MNIST dataset.

    Args:
        num_examples: Number of examples to load (default: 5).
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
        from PIL import Image  # noqa: F401
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

    # Ensure at least one example for each digit (up to num_examples)
    selected_indices: set[int] = set()
    digits_to_sample = list(range(10))[:num_examples]
    for digit in digits_to_sample:
        if digit_to_indices[digit]:
            selected_indices.add(random.choice(digit_to_indices[digit]))

    # Fill remaining slots randomly if needed
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
    """Run the HITL evaluation example with mixed text and image inputs."""
    print("=" * 60)
    print("Human-in-the-Loop (HITL) Evaluation Example")
    print("=" * 60)
    print("\nThis example demonstrates HITL evaluation functions.")
    print("During optimization, a GUI popup will appear for each evaluation.")
    print("You can review and edit the proposed output before continuing.")
    print("This example includes both TEXT and IMAGE (MNIST) examples.\n")

    # Create examples with text input
    print("Loading text examples...")
    text_examples = [
        Example(
            text="John Doe, 30 years old, lives in New York. Contact: john.doe@example.com",
            expected_output={
                "name": "John Doe",
                "email": "john.doe@example.com",
                "age": 30,
                "city": "New York",
            },
        ),
        Example(
            text="Jane Smith is 25 and resides in San Francisco. Email: jane.smith@email.com",
            expected_output={
                "name": "Jane Smith",
                "email": "jane.smith@email.com",
                "age": 25,
                "city": "San Francisco",
            },
        ),
    ]

    # Load MNIST image examples
    print("Loading MNIST image examples...")
    try:
        image_examples = load_mnist_examples(num_examples=3, split="train")
        print(f"Loaded {len(image_examples)} MNIST image examples")
    except ImportError as e:
        print(f"Warning: Could not load MNIST examples: {e}")
        print("Skipping image examples. Install datasets and pillow to include images.")
        image_examples = []

    # Example 1: Text examples with exact-hitl
    print("\n" + "=" * 60)
    print("Example 1: Text Examples with 'exact-hitl' evaluation")
    print("-" * 60)
    print("Score: 0.0 if you edit the output, 1.0 if you don't edit\n")

    # optimizer_text = PydanticOptimizer(
    #     model=UserProfile,
    #     examples=text_examples,
    #     evaluate_fn="exact-hitl",
    #     model_id="gpt-4o-mini",
    #     verbose=True,
    # )

    # print("Starting optimization with text examples...")
    # print("A popup will appear for each evaluation. Review and edit as needed.\n")

    # # result_text = optimizer_text.optimize()

    # print("\n" + "=" * 60)
    # print("Results (Text Examples - exact-hitl)")
    # print("=" * 60)
    # print(f"Baseline score: {result_text.baseline_score:.2%}")
    # print(f"Optimized score: {result_text.optimized_score:.2%}")
    # print(f"Improvement: {result_text.metrics['improvement']:+.2%}")

    # Example 2: Image examples with exact-hitl (if available)
    if image_examples:
        print("\n\n" + "=" * 60)
        print("Example 2: MNIST Image Examples with 'exact-hitl' evaluation")
        print("-" * 60)
        print("Score: 0.0 if you edit the output, 1.0 if you don't edit")
        print("The popup will show the handwritten digit image.\n")

        optimizer_images = PydanticOptimizer(
            model=DigitClassification,
            examples=image_examples,
            evaluate_fn="exact-hitl",
            model_id="gpt-4o-mini",
            verbose=True,
            system_prompt=(
                "You are an expert image classification assistant specializing in handwritten "
                "digit recognition. You can accurately identify digits from 0 to 9."
            ),
            instruction_prompt=(
                "Analyze the provided handwritten digit image and identify the digit shown. "
                "The digit will be a single number from 0 to 9."
            ),
        )

        print("Starting optimization with MNIST image examples...")
        print("A popup will appear for each evaluation showing the digit image.")
        print("Review and edit the proposed digit classification as needed.\n")

        result_images = optimizer_images.optimize()

        print("\n" + "=" * 60)
        print("Results (Image Examples - exact-hitl)")
        print("=" * 60)
        print(f"Baseline score: {result_images.baseline_score:.2%}")
        print(f"Optimized score: {result_images.optimized_score:.2%}")
        print(f"Improvement: {result_images.metrics['improvement']:+.2%}")

    # Example 3: Mixed examples with levenshtein-hitl
    print("\n\n" + "=" * 60)
    print("Example 3: Mixed Text Examples with 'levenshtein-hitl' evaluation")
    print("-" * 60)
    print(
        "Score: Levenshtein similarity if you edit the output, "
        "1.0 if you don't edit\n"
    )

    optimizer_levenshtein = PydanticOptimizer(
        model=UserProfile,
        examples=text_examples,
        evaluate_fn="levenshtein-hitl",
        model_id="gpt-4o-mini",
        verbose=True,
    )

    print("Starting optimization with levenshtein-hitl evaluation...")
    print("A popup will appear for each evaluation. Review and edit as needed.\n")

    result_levenshtein = optimizer_levenshtein.optimize()

    print("\n" + "=" * 60)
    print("Results (levenshtein-hitl)")
    print("=" * 60)
    print(f"Baseline score: {result_levenshtein.baseline_score:.2%}")
    print(f"Optimized score: {result_levenshtein.optimized_score:.2%}")
    print(f"Improvement: {result_levenshtein.metrics['improvement']:+.2%}")

    print("\n" + "=" * 60)
    print("Optimized Descriptions (Text Examples)")
    print("=" * 60)
    for field_path, description in result_text.optimized_descriptions.items():
        print(f"\n{field_path}:")
        print(f"  {description}")

    if image_examples:
        print("\n" + "=" * 60)
        print("Optimized Descriptions (Image Examples)")
        print("=" * 60)
        for field_path, description in result_images.optimized_descriptions.items():
            print(f"\n{field_path}:")
            print(f"  {description}")


if __name__ == "__main__":
    main()

