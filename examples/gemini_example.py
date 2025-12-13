"""Example demonstrating DSPydantic with Google Gemini models.

This example shows how to use Google's Gemini models for optimizing
Pydantic model field descriptions. Gemini models are great for
multimodal tasks and long-context understanding.

To run this example:
1. Set your Google API key: export GOOGLE_API_KEY="your-key"
2. Install dependencies: pip install dspydantic
3. Run: python examples/gemini_example.py
"""

from typing import Literal

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer, create_optimized_model


class ProductReview(BaseModel):
    """Product review analysis model."""

    product_name: str = Field(description="Name of the product")
    sentiment: Literal["positive", "negative", "mixed"] = Field(
        description="Overall sentiment of the review"
    )
    rating: int = Field(description="Rating from 1 to 5")
    pros: list[str] = Field(
        default_factory=list,
        description="Positive aspects mentioned"
    )
    cons: list[str] = Field(
        default_factory=list,
        description="Negative aspects mentioned"
    )
    would_recommend: bool = Field(description="Whether the reviewer would recommend")


def main():
    """Run the Gemini example."""
    # Create examples
    examples = [
        Example(
            text=(
                "Just got the Sony WH-1000XM5 headphones and I'm blown away! "
                "The noise cancellation is incredible - you can't hear anything around you. "
                "Sound quality is amazing, very crisp and clear. Battery lasts forever, "
                "easily 2-3 days of use. Only downside is they're a bit pricey at $400, "
                "but worth every penny. Would definitely recommend! 5 stars."
            ),
            expected_output=ProductReview(
                product_name="Sony WH-1000XM5",
                sentiment="positive",
                rating=5,
                pros=[
                    "Incredible noise cancellation",
                    "Amazing sound quality",
                    "Long battery life"
                ],
                cons=["Expensive price"],
                would_recommend=True
            )
        ),
        Example(
            text=(
                "The Samsung Galaxy Buds2 Pro are decent but not great. "
                "They fit well and the active noise cancellation works okay for the price. "
                "However, the sound quality is just average - lacks bass. "
                "Battery life is disappointing at only 5 hours. "
                "For $230, I expected more. Might look at other options. 3/5 stars."
            ),
            expected_output=ProductReview(
                product_name="Samsung Galaxy Buds2 Pro",
                sentiment="mixed",
                rating=3,
                pros=[
                    "Good fit",
                    "Decent noise cancellation for price"
                ],
                cons=[
                    "Average sound quality",
                    "Poor battery life"
                ],
                would_recommend=False
            )
        ),
        Example(
            text=(
                "Do NOT buy the Beats Studio Buds. Total waste of money. "
                "They don't stay in your ears, fall out constantly during workouts. "
                "Sound is muffled and tinny. Noise cancellation barely works. "
                "Battery died after 3 hours. Customer service was unhelpful. "
                "Returning them ASAP. 1 star and that's being generous."
            ),
            expected_output=ProductReview(
                product_name="Beats Studio Buds",
                sentiment="negative",
                rating=1,
                pros=[],
                cons=[
                    "Poor fit",
                    "Bad sound quality",
                    "Ineffective noise cancellation",
                    "Short battery life",
                    "Unhelpful customer service"
                ],
                would_recommend=False
            )
        ),
        Example(
            text=(
                "Apple AirPods Pro (2nd gen) are fantastic! Best earbuds I've owned. "
                "Noise cancellation is top-tier, blocks everything. "
                "Transparency mode is really natural. Great for calls. "
                "Spatial audio is mind-blowing for movies. "
                "The case with MagSafe is super convenient. "
                "Only con is the price at $250, but Apple quality. "
                "Highly recommend to anyone with an iPhone. 5/5."
            ),
            expected_output=ProductReview(
                product_name="Apple AirPods Pro (2nd gen)",
                sentiment="positive",
                rating=5,
                pros=[
                    "Excellent noise cancellation",
                    "Natural transparency mode",
                    "Great for calls",
                    "Amazing spatial audio",
                    "Convenient MagSafe case"
                ],
                cons=["High price"],
                would_recommend=True
            )
        ),
    ]

    print("=" * 60)
    print("DSPydantic with Google Gemini Example")
    print("=" * 60)
    print()

    # Create optimizer with Gemini model
    optimizer = PydanticOptimizer(
        model=ProductReview,
        examples=examples,
        model_id="gemini/gemini-2.5-flash-lite",
        verbose=True,
        num_threads=2
    )

    print("Starting optimization with Google Gemini...")
    print("Model: gemini/gemini-1.5-pro")
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
    OptimizedProductReview = create_optimized_model(
        ProductReview,
        result.optimized_descriptions
    )

    print("\n" + "=" * 60)
    print("Using the Optimized Model")
    print("=" * 60)
    print()
    print("You can now use OptimizedProductReview with any LLM provider:")
    print()
    print("Example with OpenAI:")
    print("""
from openai import OpenAI

client = OpenAI()
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": review_text}],
    response_format=OptimizedProductReview
)
review = response.choices[0].message.parsed
""")
    print()
    print("Example with Anthropic Claude:")
    print("""
import anthropic
import instructor

client = instructor.from_anthropic(anthropic.Anthropic())
review = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": review_text}],
    response_model=OptimizedProductReview
)
""")
    print()
    print("Example with Google Gemini:")
    print("""
import google.generativeai as genai
import instructor

client = instructor.from_gemini(
    client=genai.GenerativeModel(model_name="gemini-1.5-pro")
)
review = client.messages.create(
    messages=[{"role": "user", "content": review_text}],
    response_model=OptimizedProductReview
)
""")


if __name__ == "__main__":
    main()
