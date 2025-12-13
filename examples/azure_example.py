"""Example demonstrating DSPydantic with Azure OpenAI.

This example shows how to use Azure OpenAI for optimizing
Pydantic model field descriptions. Azure OpenAI provides
enterprise-grade deployment with enhanced security and compliance.

To run this example:
1. Set your Azure OpenAI credentials:
   export AZURE_OPENAI_API_KEY="your-key"
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
   export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
2. Install dependencies: pip install dspydantic
3. Run: python examples/azure_example.py
"""

import os
from typing import Literal

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer, create_optimized_model


class MedicalRecord(BaseModel):
    """Medical record extraction model."""

    patient_id: str = Field(description="Patient identifier")
    diagnosis: str = Field(description="Primary diagnosis")
    severity: Literal["mild", "moderate", "severe", "critical"] = Field(
        description="Severity level"
    )
    symptoms: list[str] = Field(
        default_factory=list,
        description="Reported symptoms"
    )
    prescribed_treatment: str = Field(description="Treatment plan")
    follow_up_required: bool = Field(description="Whether follow-up is needed")


def main():
    """Run the Azure OpenAI example."""
    # Create examples
    examples = [
        Example(
            text=(
                "Patient #A12345 presents with seasonal allergies. "
                "Symptoms include sneezing, runny nose, and itchy eyes. "
                "Severity is mild. Prescribed antihistamines (Claritin 10mg daily). "
                "No follow-up needed unless symptoms worsen."
            ),
            expected_output=MedicalRecord(
                patient_id="A12345",
                diagnosis="Seasonal allergies",
                severity="mild",
                symptoms=["sneezing", "runny nose", "itchy eyes"],
                prescribed_treatment="Antihistamines (Claritin 10mg daily)",
                follow_up_required=False
            )
        ),
        Example(
            text=(
                "Patient B67890 diagnosed with Type 2 Diabetes. "
                "Moderate severity with elevated blood sugar levels. "
                "Patient reports increased thirst, frequent urination, and fatigue. "
                "Starting metformin 500mg twice daily and lifestyle modifications. "
                "Schedule follow-up in 3 months to monitor glucose levels."
            ),
            expected_output=MedicalRecord(
                patient_id="B67890",
                diagnosis="Type 2 Diabetes",
                severity="moderate",
                symptoms=["increased thirst", "frequent urination", "fatigue"],
                prescribed_treatment="Metformin 500mg twice daily and lifestyle modifications",
                follow_up_required=True
            )
        ),
        Example(
            text=(
                "Patient C24680 admitted with acute pneumonia. "
                "Severe presentation with high fever, chest pain, and difficulty breathing. "
                "Started on IV antibiotics (Ceftriaxone) and oxygen therapy. "
                "Hospitalization required. Follow-up with pulmonologist in 2 weeks."
            ),
            expected_output=MedicalRecord(
                patient_id="C24680",
                diagnosis="Acute pneumonia",
                severity="severe",
                symptoms=["high fever", "chest pain", "difficulty breathing"],
                prescribed_treatment="IV antibiotics (Ceftriaxone) and oxygen therapy",
                follow_up_required=True
            )
        ),
    ]

    print("=" * 60)
    print("DSPydantic with Azure OpenAI Example")
    print("=" * 60)
    print()

    # Get Azure credentials from environment
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not azure_endpoint:
        print("Error: AZURE_OPENAI_ENDPOINT environment variable not set")
        print("Please set:")
        print('  export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"')
        print('  export AZURE_OPENAI_API_KEY="your-key"')
        print('  export AZURE_OPENAI_API_VERSION="2024-02-15-preview"')
        return

    # Create optimizer with Azure OpenAI
    optimizer = PydanticOptimizer(
        model=MedicalRecord,
        examples=examples,
        model_id="azure/gpt-4o",  # or "azure/gpt-4-turbo", "azure/gpt-35-turbo"
        api_base=azure_endpoint,
        api_version=azure_api_version,
        # API key will be read from AZURE_OPENAI_API_KEY env var
        verbose=True,
        num_threads=2
    )

    print("Starting optimization with Azure OpenAI...")
    print(f"Model: azure/gpt-4o")
    print(f"Endpoint: {azure_endpoint}")
    print(f"API Version: {azure_api_version}")
    print()

    # Run optimization
    result = optimizer.optimize()

    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print()

    print("Optimized Field Descriptions:")
    for field_name, description in result.optimized_descriptions.items():
        print(f"  {field_name}: {description}")

    print()
    print("Optimization Metrics:")
    print(f"  Baseline Score: {result.baseline_score:.2f}")
    print(f"  Optimized Score: {result.optimized_score:.2f}")
    print(f"  Improvement: {(result.optimized_score - result.baseline_score):.2f}")

    # Create optimized model
    OptimizedMedicalRecord = create_optimized_model(
        MedicalRecord,
        result.optimized_descriptions
    )

    print("\n" + "=" * 60)
    print("Using with Azure OpenAI")
    print("=" * 60)
    print()
    print("Now you can use OptimizedMedicalRecord with Azure OpenAI:")
    print("""
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

response = client.beta.chat.completions.parse(
    model="gpt-4o",  # Your Azure deployment name
    messages=[{"role": "user", "content": medical_text}],
    response_format=OptimizedMedicalRecord
)
record = response.choices[0].message.parsed
""")
    print()
    print("Benefits of Azure OpenAI:")
    print("  • Enterprise-grade security and compliance")
    print("  • Data residency in your region")
    print("  • Private networking with VNet support")
    print("  • SLA-backed availability")
    print("  • Content filtering and responsible AI")


if __name__ == "__main__":
    main()
