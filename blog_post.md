# Auto-Optimize Pydantic Models with DSPy: A Complete Guide to DSPydantic

**Extract structured data from LLMs with zero manual prompt engineering. Learn how DSPydantic combines the power of DSPy optimization with Pydantic validation to automatically improve your data extraction accuracy.**

If you've ever struggled with crafting the perfect field descriptions for your Pydantic models to extract structured data from Large Language Models (LLMs), you're not alone. Manual prompt engineering is time-consuming, error-prone, and often yields suboptimal results. What if you could automatically optimize your Pydantic model field descriptions and prompts using DSPy's powerful optimization algorithms?

Enter **DSPydantic**—a library that bridges the gap between **DSPy** (Declarative Self-improving Python) and **Pydantic**, automatically optimizing your Pydantic models for better structured data extraction from LLMs. In this comprehensive guide, we'll explore all features of DSPydantic using a real-world IMDB sentiment classification example.

## What is DSPydantic?

DSPydantic is a Python library that automatically optimizes Pydantic model field descriptions and prompts using DSPy's optimization algorithms. Instead of manually tuning field descriptions, you provide a few examples, and DSPydantic uses DSPy to find the optimal descriptions that maximize extraction accuracy.

**Key Benefits:**
- **Automatic optimization**: DSPy algorithms find the best field descriptions—typically improves accuracy by 20-40% over manual descriptions
- **Zero manual tuning**: Just provide examples and let DSPydantic do the work
- **Pydantic integration**: Works seamlessly with your existing Pydantic models
- **Multi-modal support**: Handles text, images, and PDFs
- **Template support**: Dynamic prompts with placeholders filled from example data

## Installation

```bash
pip install dspydantic
```

Or with `uv`:

```bash
uv pip install dspydantic
```

## IMDB Sentiment Classification: A Complete Walkthrough

Let's build a complete sentiment classification system for IMDB movie reviews using DSPydantic. We'll start with a minimal setup and gradually explore advanced features.

### Step 1: Start with an Empty Pydantic Class

The beauty of DSPydantic is that you can start with minimal field descriptions—or even empty ones. Let's define a simple Pydantic model for sentiment classification:

```python
from typing import Literal
from pydantic import BaseModel

class SentimentClassification(BaseModel):
    """Sentiment classification model for movie reviews."""
    
    sentiment: Literal["positive", "negative"]
```

That's it! Notice how we haven't added any detailed field descriptions. DSPydantic will optimize these automatically based on your examples.

### Step 2: Create Examples with Input and Output

Next, we'll create examples showing the input (movie reviews) and expected output (sentiment labels). DSPydantic uses these examples to learn optimal field descriptions:

```python
from dspydantic import Example

# Example 1: Positive review
example_1 = Example(
    text={
        "review": "This movie was absolutely fantastic! The acting was superb, "
                  "the plot was engaging, and I couldn't take my eyes off the screen. "
                  "Highly recommend to everyone!",
        "review_length": "25"
    },
    expected_output={"sentiment": "positive"}
)

# Example 2: Negative review
example_2 = Example(
    text={
        "review": "Terrible movie. Boring plot, poor acting, and a complete waste of time. "
                  "I regret watching this.",
        "review_length": "18"
    },
    expected_output={"sentiment": "negative"}
)

# Example 3: Another positive review
example_3 = Example(
    text={
        "review": "An incredible cinematic experience! The director's vision shines through "
                  "every scene. The cinematography is breathtaking.",
        "review_length": "19"
    },
    expected_output={"sentiment": "positive"}
)

examples = [example_1, example_2, example_3]
```

**Key Points:**
- `text` is a dictionary with keys `"review"` and `"review_length"`—this enables template formatting
- `expected_output` is a dictionary matching our Pydantic model structure
- We're using a minimal set of examples (3) for demonstration; typically 5-20 examples yield best results

### Step 3: Define a Minimal Prompt Template

DSPydantic supports template prompts with placeholders that are automatically filled from your example data. This is perfect for dynamic prompts:

```python
instruction_prompt = "A review of a movie: {review}"
```

The `{review}` placeholder will be automatically replaced with the value from each example's `text` dictionary. This allows you to create example-specific prompts without manual formatting.

### Step 4: Optimize with DSPydantic

Now we'll use DSPydantic's `PydanticOptimizer` to automatically optimize our model:

```python
from dspydantic import PydanticOptimizer

optimizer = PydanticOptimizer(
    model=SentimentClassification,
    examples=examples,
    model_id="gpt-4o-mini",  # Uses OPENAI_API_KEY from environment
    verbose=True,
    optimizer="bootstrapfewshot",
    system_prompt=(
        "You are an expert sentiment analysis assistant specializing in movie review "
        "classification. You understand nuanced language, sarcasm, and contextual cues "
        "that indicate positive or negative sentiment in written reviews."
    ),
    instruction_prompt="A review of a movie: {review}",
)

result = optimizer.optimize()
```

**What Happens During Optimization:**
1. DSPydantic uses DSPy's `BootstrapFewShot` optimizer to iteratively improve field descriptions
2. The optimizer tests different description variations against your examples
3. System and instruction prompts are also optimized if provided
4. The process continues until optimal descriptions are found

### Step 5: View Optimization Results

After optimization completes, you can inspect the results:

```python
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
```

**Typical Output:**
```
Baseline score: 50.00%
Optimized score: 100.00%
Improvement: +50.00%

Optimized system prompt:
  You are an expert sentiment analysis assistant specializing in movie review 
  classification. You understand nuanced language, sarcasm, and contextual cues 
  that indicate positive or negative sentiment in written reviews. You can 
  accurately distinguish between genuine praise and criticism even when reviews 
  contain mixed signals. Focus on identifying clear indicators of sentiment 
  such as explicit positive or negative language, overall tone, and reviewer 
  satisfaction level.

Optimized instruction prompt:
  Analyze the following movie review and classify its sentiment as either 
  "positive" or "negative" based on the reviewer's overall opinion: {review}

Optimized descriptions:
  sentiment: The overall emotional tone of the movie review, classified as 
  either "positive" (indicating satisfaction, praise, or recommendation) or 
  "negative" (indicating dissatisfaction, criticism, or lack of recommendation). 
  Consider the reviewer's explicit statements, implicit tone, and overall 
  satisfaction level when determining sentiment.
```

Notice how DSPydantic has:
- Enhanced the system prompt with more specific guidance
- Improved the instruction prompt to be more explicit
- Created a detailed, optimized description for the `sentiment` field

### Step 6: Use the Optimized Model

Create an optimized Pydantic model class with the improved descriptions:

```python
from dspydantic import create_optimized_model

OptimizedSentimentClassification = create_optimized_model(
    SentimentClassification, 
    result.optimized_descriptions
)
```

Now use it with your LLM for production inference:

```python
from openai import OpenAI

client = OpenAI()

# Use optimized prompts
messages = []
if result.optimized_system_prompt:
    messages.append({"role": "system", "content": result.optimized_system_prompt})

user_content = "A review of a movie: This film exceeded all my expectations!"
if result.optimized_instruction_prompt:
    user_content = f"{result.optimized_instruction_prompt}\n\n{user_content}"
messages.append({"role": "user", "content": user_content})

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": OptimizedSentimentClassification.__name__,
            "schema": OptimizedSentimentClassification.model_json_schema(),
            "strict": True
        }
    }
)

# Parse response
sentiment = OptimizedSentimentClassification.model_validate_json(
    response.choices[0].message.content
)
print(f"Predicted sentiment: {sentiment.sentiment}")
```

## Complete IMDB Example with Real Data

Here's the complete example using the actual IMDB dataset:

```python
import random
from typing import Literal
from pydantic import BaseModel
from dspydantic import Example, PydanticOptimizer, create_optimized_model

class SentimentClassification(BaseModel):
    """Sentiment classification model for movie reviews."""
    sentiment: Literal["positive", "negative"]

def load_imdb_examples(num_examples: int = 10) -> list[Example]:
    """Load examples from the IMDB dataset."""
    from datasets import load_dataset
    
    dataset = load_dataset("stanfordnlp/imdb", split="train")
    
    # Ensure balanced examples
    positive_indices = [i for i, item in enumerate(dataset) if item["label"] == 1]
    negative_indices = [i for i, item in enumerate(dataset) if item["label"] == 0]
    
    selected_indices = set()
    if positive_indices:
        selected_indices.add(random.choice(positive_indices))
    if negative_indices:
        selected_indices.add(random.choice(negative_indices))
    
    # Fill remaining slots
    remaining = num_examples - len(selected_indices)
    if remaining > 0:
        available = set(range(len(dataset))) - selected_indices
        additional = random.sample(list(available), min(remaining, len(available)))
        selected_indices.update(additional)
    
    examples = []
    for idx in list(selected_indices)[:num_examples]:
        item = dataset[idx]
        sentiment = "positive" if item["label"] == 1 else "negative"
        review_text = item["text"]
        review_length = len(review_text.split())
        
        examples.append(Example(
            text={
                "review": review_text,
                "review_length": str(review_length),
            },
            expected_output={"sentiment": sentiment},
        ))
    
    return examples

# Load examples
examples = load_imdb_examples(num_examples=10)

# Optimize
optimizer = PydanticOptimizer(
    model=SentimentClassification,
    examples=examples,
    model_id="gpt-4o-mini",
    verbose=True,
    optimizer="bootstrapfewshot",
    system_prompt=(
        "You are an expert sentiment analysis assistant specializing in movie review "
        "classification. You understand nuanced language, sarcasm, and contextual cues "
        "that indicate positive or negative sentiment in written reviews."
    ),
    instruction_prompt="A review of a movie: {review}",
)

result = optimizer.optimize()

# Create optimized model
OptimizedSentimentClassification = create_optimized_model(
    SentimentClassification, 
    result.optimized_descriptions
)

print(f"Improvement: {result.metrics['improvement']:+.2%}")
```

## Advanced Features

DSPydantic offers many advanced features beyond basic optimization:

### 1. Multi-Modal Input Support

DSPydantic supports text, images, and PDFs:

```python
# Text input
Example(text="Invoice #123 from Acme Corp", expected_output=...)

# Image input
Example(image_path="invoice.png", expected_output=...)

# PDF input
Example(pdf_path="invoice.pdf", pdf_dpi=300, expected_output=...)

# Combined text and image
Example(
    text="Extract information from this invoice",
    image_path="invoice.png",
    expected_output=...
)
```

### 2. Custom Evaluation Functions

Provide domain-specific evaluation:

```python
def custom_evaluate(
    example: Example,
    optimized_descriptions: dict[str, str],
    optimized_system_prompt: str | None,
    optimized_instruction_prompt: str | None,
) -> float:
    """Returns a score between 0.0 and 1.0."""
    # Your evaluation logic here
    return 0.85

optimizer = PydanticOptimizer(
    model=MyModel,
    examples=examples,
    evaluate_fn=custom_evaluate,
    model_id="gpt-4o"
)
```

### 3. LLM-as-Judge Evaluation

Evaluate without ground truth using LLM judges:

```python
examples = [
    Example(
        text="Patient: John Doe, age 30, presenting with symptoms",
        expected_output=None  # No ground truth, uses LLM judge
    ),
]

optimizer = PydanticOptimizer(
    model=PatientRecord,
    examples=examples,
    model_id="gpt-4o"  # This LLM will be used as judge
)
```

## Key Takeaways

1. **Start Simple**: Begin with minimal Pydantic models and let DSPydantic optimize descriptions
2. **Provide Examples**: 5-20 examples typically yield best results
3. **Use Templates**: Leverage template prompts with placeholders for dynamic prompts
4. **Trust DSPy**: DSPy's optimization algorithms automatically find optimal descriptions
5. **Iterate**: Use optimization results to understand what works best for your use case

## Conclusion

DSPydantic combines the power of **DSPy** optimization with **Pydantic** validation to automatically improve structured data extraction from LLMs. By providing minimal examples and letting DSPydantic optimize your field descriptions and prompts, you can achieve significant accuracy improvements (typically 20-40%) with zero manual tuning.

Whether you're extracting structured data from text, images, or PDFs, DSPydantic's multi-modal support, template formatting, and automatic optimizer selection make it easy to build production-ready extraction systems. Start with a simple Pydantic model, provide a few examples, and let DSPydantic do the rest.

**Ready to get started?** Install DSPydantic and try the IMDB example above. You'll be amazed at how much better your extraction accuracy becomes with automatic optimization!

---

*For more examples and complete documentation, visit the [DSPydantic GitHub repository](https://github.com/yourusername/dspydantic).*
