"""Named Entity Recognition (NER) example - Medical entities.

This example demonstrates how to optimize a Pydantic model for extracting named entities
from medical text, based on GLiNER2's entity extraction tutorial example.
It extracts medication, dosage, symptom, and time entities from patient records.
"""

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


class MedicalEntities(BaseModel):
    """Medical entity recognition model for patient records."""

    medication: list[str] = Field(
        default_factory=list,
        description="Names of drugs, medications, or pharmaceutical substances",
    )
    dosage: list[str] = Field(
        default_factory=list,
        description="Specific amounts like '400mg', '2 tablets', or '5ml'",
    )
    symptom: list[str] = Field(
        default_factory=list,
        description="Medical symptoms, conditions, or patient complaints",
    )
    time: list[str] = Field(
        default_factory=list,
        description="Time references like '2 PM', 'morning', or 'after lunch'",
    )


def main():
    """Run the medical entity recognition optimization example."""
    print("=" * 60)
    print("Named Entity Recognition (NER) Example - Medical Entities")
    print("=" * 60)
    print("\nThis example extracts medical entities from patient records:")
    print("- Medication: Names of drugs, medications, or pharmaceutical substances")
    print("- Dosage: Specific amounts like '400mg', '2 tablets', or '5ml'")
    print("- Symptom: Medical symptoms, conditions, or patient complaints")
    print("- Time: Time references like '2 PM', 'morning', or 'after lunch'\n")

    # Create examples based on GLiNER2 tutorial
    examples = [
        Example(
            text="Patient received 400mg ibuprofen for severe headache at 2 PM.",
            expected_output=MedicalEntities(
                medication=["ibuprofen"],
                dosage=["400mg"],
                symptom=["severe headache"],
                time=["2 PM"],
            ),
        ),
        Example(
            text="Prescribed 2 tablets of acetaminophen for fever this morning.",
            expected_output=MedicalEntities(
                medication=["acetaminophen"],
                dosage=["2 tablets"],
                symptom=["fever"],
                time=["this morning"],
            ),
        ),
        Example(
            text="Administered 5ml of cough syrup for persistent cough after lunch.",
            expected_output=MedicalEntities(
                medication=["cough syrup"],
                dosage=["5ml"],
                symptom=["persistent cough"],
                time=["after lunch"],
            ),
        ),
        Example(
            text="Patient took 500mg aspirin for chest pain at 3:30 PM.",
            expected_output=MedicalEntities(
                medication=["aspirin"],
                dosage=["500mg"],
                symptom=["chest pain"],
                time=["3:30 PM"],
            ),
        ),
        Example(
            text="Given 1 tablet of lorazepam for anxiety in the evening.",
            expected_output=MedicalEntities(
                medication=["lorazepam"],
                dosage=["1 tablet"],
                symptom=["anxiety"],
                time=["evening"],
            ),
        ),
    ]

    print(f"Created {len(examples)} examples\n")
    print("Sample examples:")
    for i, example in enumerate(examples[:2], 1):
        print(f"\nExample {i}:")
        text_preview = example.input_data.get("text", "")
        print(f"  Text: {text_preview}")
        print("  Expected entities:")
        expected = example.expected_output
        if isinstance(expected, dict):
            for key, value in expected.items():
                if value:
                    print(f"    {key}: {value}")

    # Create optimizer
    optimizer = PydanticOptimizer(
        model=MedicalEntities,
        examples=examples,
        model_id="gpt-4o-mini",
        verbose=True,
        optimizer="miprov2zeroshot",
        system_prompt=(
            "You are an expert medical entity recognition assistant specializing in "
            "extracting structured information from patient records. You can accurately "
            "identify medications, dosages, symptoms, and time references even when "
            "they appear in various forms or contexts."
        ),
        instruction_prompt=(
            "Extract all medical entities from the provided patient record text. "
            "Identify medications, dosages, symptoms, and time references mentioned in the text. "
            "Return the extracted entities organized by category in the JSON schema format."
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
    print("\nOptimized field descriptions:")
    for field_path, description in result.optimized_descriptions.items():
        print(f"\n  {field_path}:")
        print(f"    {description}")


if __name__ == "__main__":
    main()
