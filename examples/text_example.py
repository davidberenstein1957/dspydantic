"""PetEVAL veterinary EHR extraction example using SAVSNET/PetEVAL dataset.

This example demonstrates how to optimize a Pydantic model for extracting structured
information from veterinary electronic health records (EHRs) from the PetEVAL dataset.
"""

import ast
import random
from typing import Any

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


class VeterinaryRecord(BaseModel):
    """Veterinary record extraction model for PetEVAL clinical narratives."""

    diseases: list[str] = Field(
        default_factory=list,
        description="Recorded disease entities within the document",
    )
    icd_labels: list[str] = Field(
        default_factory=list,
        description=(
            "A list of ICD-11 syndromic chapter labels (International Classification of Diseases, 11th Revision) "
            "that categorize diagnoses and clinical syndromes in this record. These standardized codes help organize, "
            "record, and report health data to improve diagnosis accuracy and patient care."
        ),
    )
    anonymised_entities: list[str] = Field(
        default_factory=list,
        description=("NER tags for anonymization applied in the record."),
    )


def _extract_item_labels(item: dict[str, Any], text: str) -> dict[str, Any]:
    """Extract labels from a dataset item.

    Args:
        item: Dataset item dictionary.
        text: Clinical narrative text.

    Returns:
        Dictionary with extracted labels.
    """
    expected_output: dict[str, Any] = {
        "diseases": [],
        "icd_labels": [],
        "anonymised_entities": [],
    }

    # Extract ICD-11 labels (multilabel)
    if "icd_label" in item:
        icd_labels = item["icd_label"]
        if isinstance(icd_labels, list):
            expected_output["icd_labels"] = icd_labels

    # Extract disease entities from NER tags
    # Note: "disease" field contains Python literal string representation
    if "disease" in item:
        disease_str = item["disease"]
        if isinstance(disease_str, str) and disease_str != "[]":
            try:
                diseases = ast.literal_eval(disease_str)
                if isinstance(diseases, list):
                    # Extract entity text from the text using start/end positions
                    disease_entities = []
                    for disease_tag in diseases:
                        if isinstance(disease_tag, dict):
                            start = disease_tag.get("start", 0)
                            end = disease_tag.get("end", 0)
                            entity = disease_tag.get("entity", "")
                            # Extract the actual text from the clinical narrative
                            if start < len(text) and end <= len(text):
                                entity_text = text[start:end]
                                if entity_text:
                                    disease_entities.append(entity_text)
                                elif entity:  # Fallback to entity label
                                    disease_entities.append(entity)
                    expected_output["diseases"] = disease_entities
            except (ValueError, SyntaxError):
                # If parsing fails, skip this field
                pass

    # Extract anonymisation entities
    # Note: "annonymisation" field (double 'n') contains Python literal string
    if "annonymisation" in item:
        annonymisation_str = item["annonymisation"]
        if isinstance(annonymisation_str, str) and annonymisation_str != "[]":
            try:
                anonymisation = ast.literal_eval(annonymisation_str)
                if isinstance(anonymisation, list):
                    anonymised_entities = []
                    for anon_tag in anonymisation:
                        if isinstance(anon_tag, dict):
                            entity = anon_tag.get("entity", "")
                            # Collect anonymised entity types
                            if entity:
                                anonymised_entities.append(entity)

                    expected_output["anonymised_entities"] = anonymised_entities
            except (ValueError, SyntaxError):
                # If parsing fails, skip this field
                pass

    return expected_output


def load_peteval_examples(num_examples: int = 10) -> list[Example]:
    """Load examples from the PetEVAL dataset.

    Ensures at least one example for each unique ICD label and anonymised entity type.

    Args:
        num_examples: Number of examples to load (default: 10).

    Returns:
        List of Example objects with clinical narratives and expected structured data.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library is required. Install it with: uv pip install datasets")

    # Load the PetEVAL dataset
    dataset = load_dataset("SAVSNET/PetEVAL", split="test")
    dataset_size = len(dataset)

    # First pass: collect all unique labels and find examples for each
    all_icd_labels: set[str] = set()
    all_anonymised_entities: set[str] = set()
    label_to_indices: dict[str, list[int]] = {}  # Maps label to indices containing it

    for idx in range(dataset_size):
        item = dataset[idx]
        text = item.get("sentence", "")
        labels = _extract_item_labels(item, text)

        # Track ICD labels
        for icd_label in labels["icd_labels"]:
            all_icd_labels.add(icd_label)
            if icd_label not in label_to_indices:
                label_to_indices[icd_label] = []
            label_to_indices[icd_label].append(idx)

        # Track anonymised entities
        for anon_entity in labels["anonymised_entities"]:
            all_anonymised_entities.add(anon_entity)
            if anon_entity not in label_to_indices:
                label_to_indices[anon_entity] = []
            label_to_indices[anon_entity].append(idx)

    # Collect indices to ensure label coverage
    selected_indices: set[int] = set()

    # Ensure at least one example for each ICD label
    for icd_label in all_icd_labels:
        if icd_label in label_to_indices and label_to_indices[icd_label]:
            selected_indices.add(random.choice(label_to_indices[icd_label]))

    # Ensure at least one example for each anonymised entity type
    for anon_entity in all_anonymised_entities:
        if anon_entity in label_to_indices and label_to_indices[anon_entity]:
            selected_indices.add(random.choice(label_to_indices[anon_entity]))

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
        text = item.get("sentence", "")
        expected_output = _extract_item_labels(item, text)

        example = Example(
            text=text,
            expected_output=expected_output,
        )
        examples.append(example)

    return examples


def main():
    """Run the PetEVAL veterinary record extraction optimization example."""
    print("Loading PetEVAL dataset examples...")
    examples = load_peteval_examples(num_examples=10)

    print(f"Loaded {len(examples)} examples")
    print("\nSample examples:")
    for i, example in enumerate(examples[:3], 1):
        print(f"\nExample {i}:")
        text_preview = example.input_data.get("text", "")[:200]
        print(f"  Text preview: {text_preview}...")
        print(f"  Expected output keys: {list(example.expected_output.keys())}")

    # Create optimizer with system and instruction prompts
    optimizer = PydanticOptimizer(
        model=VeterinaryRecord,
        examples=examples,
        model_id="gpt-4o-mini",  # Will use OPENAI_API_KEY from environment
        verbose=True,
        optimizer="miprov2zeroshot",
        system_prompt=(
            "You are an expert veterinary information extraction assistant. "
            "Your task is to extract structured medical information from veterinary "
            "electronic health records (EHRs) with high accuracy and attention to detail. "
            "You understand medical terminology, disease classifications, and clinical "
            "documentation standards."
        ),
        instruction_prompt=(
            "Extract structured information from the provided veterinary clinical narrative. "
            "Identify all disease entities mentioned in the text, classify them according to "
            "ICD-11 syndromic chapter labels, and identify any anonymised entities that should "
            "be protected. Return the extracted information in the exact format specified by "
            "the JSON schema, ensuring all fields are accurately populated based on the "
            "clinical text."
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
    improvement = result.metrics["improvement"]
    print(f"Improvement: {improvement:+.2%}")
    print("\nOptimized system prompt:")
    print(f"  {result.optimized_system_prompt}")
    print("\nOptimized instruction prompt:")
    print(f"  {result.optimized_instruction_prompt}")
    print("\nOptimized descriptions:")
    for field_path, description in result.optimized_descriptions.items():
        print(f"  {field_path}: {description}")


if __name__ == "__main__":
    main()
