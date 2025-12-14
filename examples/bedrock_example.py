"""Example demonstrating DSPydantic with AWS Bedrock (Claude models).

This example shows how to use AWS Bedrock with Claude models (Haiku 4.5 and Sonnet 4.5)
for optimizing Pydantic model field descriptions. AWS Bedrock provides access to
foundation models through a managed service with AWS-native security and compliance.

Supported Models:
- Claude 3.5 Haiku (us.anthropic.claude-3-5-haiku-20241022-v1:0): Fast, cost-effective
- Claude 3.5 Sonnet v2 (us.anthropic.claude-3-5-sonnet-20241022-v2:0): Most intelligent

To run this example:
1. Configure AWS credentials using one of these methods:
   a) AWS Profile (recommended):
      Configure in ~/.aws/credentials or ~/.aws/config
      Set profile: export AWS_PROFILE="your-profile-name"

   b) Environment variables:
      export AWS_ACCESS_KEY_ID="your-access-key"
      export AWS_SECRET_ACCESS_KEY="your-secret-key"
      export AWS_REGION="us-east-1"  # or your preferred region

   c) IAM role (when running on EC2/ECS/Lambda)

2. Ensure your AWS credentials have bedrock:InvokeModel permissions
3. Install dependencies: pip install dspydantic boto3
4. Run: python examples/bedrock_example.py

For more information on AWS Bedrock:
https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html
"""

import os
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer, create_optimized_model


class CustomerSupport(BaseModel):
    """Customer support ticket analysis model."""

    ticket_id: str = Field(description="Unique ticket identifier")
    category: Literal["billing", "technical", "account", "product", "shipping", "other"] = Field(
        description="Support ticket category"
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(description="Priority level")
    sentiment: Literal["positive", "neutral", "negative", "frustrated"] = Field(
        description="Customer sentiment"
    )
    issues: list[str] = Field(default_factory=list, description="List of reported issues")
    resolution_needed: bool = Field(description="Whether immediate resolution is required")
    estimated_resolution_time: str = Field(description="Expected time to resolve in hours or days")


def main():
    """Run the AWS Bedrock example."""
    # Create examples
    examples = [
        Example(
            text=(
                "Ticket #TK-12345: Customer reports they were charged twice for the same order. "
                "They placed an order for $299.99 last week and noticed two identical charges "
                "on their credit card statement. Customer is very frustrated and wants an immediate "
                "refund. This needs to be resolved within 24 hours."
            ),
            expected_output=CustomerSupport(
                ticket_id="TK-12345",
                category="billing",
                priority="high",
                sentiment="frustrated",
                issues=["Double charge", "Duplicate billing for same order"],
                resolution_needed=True,
                estimated_resolution_time="24 hours",
            ),
        ),
        Example(
            text=(
                "Ticket #TK-67890: User having trouble logging into their account. "
                "They tried password reset but didn't receive the email. Checked spam folder already. "
                "Customer is patient but needs access soon for an important meeting tomorrow. "
                "Should be fixable quickly by resending the reset email or clearing the cache."
            ),
            expected_output=CustomerSupport(
                ticket_id="TK-67890",
                category="technical",
                priority="medium",
                sentiment="neutral",
                issues=[
                    "Cannot login to account",
                    "Password reset email not received",
                ],
                resolution_needed=True,
                estimated_resolution_time="2 hours",
            ),
        ),
        Example(
            text=(
                "Ticket #TK-11111: Customer wants to update their shipping address "
                "for an order that hasn't shipped yet. Order #ORD-5678 was placed yesterday. "
                "New address is in the same city. Customer is polite and mentioned it's not urgent "
                "since estimated delivery is in 5 days. Can be processed within 1-2 business days."
            ),
            expected_output=CustomerSupport(
                ticket_id="TK-11111",
                category="shipping",
                priority="low",
                sentiment="positive",
                issues=["Address change request before shipment"],
                resolution_needed=False,
                estimated_resolution_time="48 hours",
            ),
        ),
        Example(
            text=(
                "Ticket #TK-99999: URGENT - Customer received defective product. "
                "The laptop they ordered won't turn on at all. They need it for work presentations "
                "starting tomorrow. Customer is understandably upset and needs immediate replacement "
                "or refund. This is a high-priority product issue that requires same-day action."
            ),
            expected_output=CustomerSupport(
                ticket_id="TK-99999",
                category="product",
                priority="urgent",
                sentiment="frustrated",
                issues=["Defective product", "Laptop won't power on", "Work urgency"],
                resolution_needed=True,
                estimated_resolution_time="24 hours",
            ),
        ),
    ]

    print("=" * 70)
    print("DSPydantic with AWS Bedrock (Claude Models) Example")
    print("=" * 70)
    print()

    # Check for AWS credentials
    aws_profile = os.getenv("AWS_PROFILE")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    print("AWS Configuration:")
    if aws_profile:
        print(f"  Profile: {aws_profile}")
    else:
        print("  Profile: default (or using environment variables/IAM role)")
    print(f"  Region: {aws_region}")
    print()

    # Choose model - you can switch between Haiku and Sonnet
    # Haiku: Fast and cost-effective
    # Sonnet: More intelligent and capable
    model_choice = (
        input(
            "Choose model:\n"
            "  1. Claude 3.5 Haiku (fast, cost-effective)\n"
            "  2. Claude 3.5 Sonnet v2 (most intelligent)\n"
            "Enter choice (1 or 2, default=1): "
        ).strip()
        or "1"
    )

    if model_choice == "2":
        model_id = "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        model_name = "Claude 3.5 Sonnet v2"
    else:
        model_id = "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"
        model_name = "Claude 3.5 Haiku"

    print(f"\nSelected model: {model_name}")
    print()

    # Create DSPy LM with Bedrock
    # DSPy supports Bedrock via the model_id format: bedrock/<model-id>
    # It will automatically use boto3 to connect to AWS Bedrock
    try:
        lm = dspy.LM(model=model_id, region_name=aws_region)
        print("✓ Successfully connected to AWS Bedrock")
        print()
    except Exception as e:
        print(f"✗ Error connecting to AWS Bedrock: {e}")
        print()
        print("Please ensure:")
        print("  1. AWS credentials are configured (AWS_PROFILE or access keys)")
        print("  2. Your IAM role/user has bedrock:InvokeModel permissions")
        print("  3. The Bedrock model is available in your region")
        print("  4. boto3 is installed: pip install boto3")
        return

    # Create optimizer with Bedrock LM
    optimizer = PydanticOptimizer(
        model=CustomerSupport,
        examples=examples,
        lm=lm,  # Pass the Bedrock LM directly
        verbose=True,
        num_threads=2,  # Bedrock has rate limits, use fewer threads
    )

    print("Starting optimization with AWS Bedrock...")
    print(f"Model: {model_name}")
    print(f"Model ID: {model_id}")
    print()

    # Run optimization
    result = optimizer.optimize()

    print("\n" + "=" * 70)
    print("Optimization Results")
    print("=" * 70)
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
    OptimizedCustomerSupport = create_optimized_model(
        CustomerSupport, result.optimized_descriptions
    )

    print("\n" + "=" * 70)
    print("Using the Optimized Model with AWS Bedrock")
    print("=" * 70)
    print()
    print("Now you can use OptimizedCustomerSupport with AWS Bedrock:")
    print(
        """
import boto3
import json
from pydantic import TypeAdapter

# Initialize Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'  # or your preferred region
)

# Prepare the request
ticket_text = "Your customer support ticket text here..."

# For Claude models, use the Messages API format
request_body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 2000,
    "messages": [
        {
            "role": "user",
            "content": f"Extract customer support information from this ticket:\\n\\n{ticket_text}\\n\\nRespond with JSON matching this schema: {OptimizedCustomerSupport.model_json_schema()}"
        }
    ],
    "temperature": 0.0
}

# Invoke the model
response = bedrock_runtime.invoke_model(
    modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0',
    body=json.dumps(request_body)
)

# Parse response
response_body = json.loads(response['body'].read())
extracted_text = response_body['content'][0]['text']

# Parse into Pydantic model
support_ticket = TypeAdapter(OptimizedCustomerSupport).validate_json(extracted_text)
print(support_ticket)
"""
    )
    print()
    print("Or using DSPy with Bedrock:")
    print(
        """
import dspy

# Configure DSPy with Bedrock
lm = dspy.LM(
    model='bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0',
    region_name='us-east-1'
)
dspy.configure(lm=lm)

# Use with DSPy signatures
class ExtractSupport(dspy.Signature):
    \"\"\"Extract customer support ticket information.\"\"\"
    ticket_text: str = dspy.InputField()
    support_info: OptimizedCustomerSupport = dspy.OutputField()

predictor = dspy.Predict(ExtractSupport)
result = predictor(ticket_text="Your ticket text here...")
print(result.support_info)
"""
    )
    print()
    print("Benefits of AWS Bedrock:")
    print("  • No model deployment or infrastructure management")
    print("  • AWS-native security and compliance (SOC, HIPAA, GDPR)")
    print("  • Pay-per-use pricing with no minimum commitments")
    print("  • Integration with AWS services (S3, Lambda, SageMaker)")
    print("  • Multiple model providers (Anthropic, Amazon Titan, etc.)")
    print("  • Data never leaves your AWS environment")
    print()
    print("Pricing Comparison (as of Dec 2024):")
    print("  • Claude 3.5 Haiku: $0.80/$4.00 per 1M input/output tokens")
    print("  • Claude 3.5 Sonnet v2: $3.00/$15.00 per 1M input/output tokens")


if __name__ == "__main__":
    main()
