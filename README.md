# DSPydantic

Automatically optimize Pydantic model field descriptions and prompts using DSPy. Get better structured data extraction from LLMs with less manual tuning.

## What is DSPydantic?

When building LLM applications that extract structured data, getting the right field descriptions and prompts is crucial. Instead of manually tweaking descriptions, `dspydantic` uses DSPy to automatically find the best descriptions and prompts based on your examples.

## Quick Start

```python
from pydantic import BaseModel, Field
from dspydantic import PydanticOptimizer, Example

# Define your Pydantic model
class User(BaseModel):
    name: str = Field(description="User name")
    age: int = Field(description="User age")
    email: str = Field(description="Email address")

# Provide examples with text input and expected Pydantic models
examples = [
    Example(
        text="John Doe, 30 years old, john@example.com",
        expected_output=User(name="John Doe", age=30, email="john@example.com")
    ),
    Example(
        text="Jane Smith, 25, jane.smith@email.com",
        expected_output=User(name="Jane Smith", age=25, email="jane.smith@email.com")
    ),
]

# Optimize field descriptions
optimizer = PydanticOptimizer(
    model=User,
    examples=examples,
    evaluate_fn="exact",  # Built-in exact matching
    model_id="gpt-4o",
    api_key="your-api-key",  # Or set OPENAI_API_KEY env var
)

result = optimizer.optimize()

# View optimized descriptions
print("Optimized descriptions:")
for field, description in result.optimized_descriptions.items():
    print(f"  {field}: {description}")
```

## Installation

```bash
pip install dspydantic
```

Or using `uv`:

```bash
uv pip install dspydantic
```

## Basic Usage

### 1. Define Your Pydantic Model

```python
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice ID")
    total_amount: float = Field(description="Total amount")
    date: str = Field(description="Invoice date")
```

### 2. Create Examples

Use plain text and Pydantic model instances:

```python
from dspydantic import Example

examples = [
    Example(
        text="Invoice #INV-2024-001, Total: $1,234.56, Date: 2024-01-15",
        expected_output=Invoice(
            invoice_number="INV-2024-001",
            total_amount=1234.56,
            date="2024-01-15"
        )
    ),
    Example(
        text="Invoice #INV-2024-002, Total: $567.89, Date: 2024-01-20",
        expected_output=Invoice(
            invoice_number="INV-2024-002",
            total_amount=567.89,
            date="2024-01-20"
        )
    ),
]
```

### 3. Optimize

```python
from dspydantic import PydanticOptimizer

optimizer = PydanticOptimizer(
    model=Invoice,
    examples=examples,
    instruction_prompt="Extract the invoice data from the text.",
    system_prompt="You are a helpful assistant that extracts invoice data from text.",
    evaluate_fn="exact",
    model_id="gpt-4o",
    verbose=True
)

result = optimizer.optimize()
```

### 4. Use Optimized Descriptions

```python
from dspydantic import apply_optimized_descriptions
from openai import OpenAI

# Create optimized schema
optimized_schema = apply_optimized_descriptions(Invoice, result.optimized_descriptions)

# Use with OpenAI structured outputs
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract: INV-2024-001, $1,234.56, 2024-01-15"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": Invoice.__name__,
            "schema": optimized_schema,
            "strict": True
        }
    }
)
```

## Working with Images

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example, PydanticOptimizer

class DigitClassification(BaseModel):
    digit: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = Field(
        description="The digit shown in the image (0-9)"
    )

examples = [
    Example(
        image_path="digit_5.png",
        expected_output=DigitClassification(digit=5)
    ),
    Example(
        image_path="digit_3.png",
        expected_output=DigitClassification(digit=3)
    ),
]

optimizer = PydanticOptimizer(
    model=DigitClassification,
    examples=examples,
    evaluate_fn="exact",
    model_id="gpt-4o"
)

result = optimizer.optimize()
```

## Working with PDFs

```python
examples = [
    Example(
        pdf_path="invoice_001.pdf",
        pdf_dpi=300,  # Optional, default is 300
        expected_output=Invoice(
            invoice_number="INV-2024-001",
            total_amount=1234.56,
            date="2024-01-15"
        )
    ),
]
```

## Nested Models

Nested models work automatically:

```python
class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    zip_code: str = Field(description="ZIP code")

class User(BaseModel):
    name: str = Field(description="User name")
    address: Address = Field(description="User address")

examples = [
    Example(
        text="John Doe, 123 Main St, New York, 10001",
        expected_output=User(
            name="John Doe",
            address=Address(street="123 Main St", city="New York", zip_code="10001")
        )
    ),
]
```

Field paths will automatically be: `"name"`, `"address.street"`, `"address.city"`, `"address.zip_code"`.

## Custom Evaluation

You can provide your own evaluation function:

```python
def evaluate(
    example: Example,
    optimized_descriptions: dict[str, str],
    optimized_system_prompt: str | None,
    optimized_instruction_prompt: str | None,
) -> float:
    """
    Evaluate how well the optimized prompts work.
    
    Returns a score between 0.0 and 1.0.
    """
    # Your evaluation logic here
    # Use optimized_descriptions and prompts with your LLM
    # Compare results with example.expected_output
    return 0.85

optimizer = PydanticOptimizer(
    model=User,
    examples=examples,
    evaluate_fn=evaluate,
    model_id="gpt-4o"
)
```

## Optimizing Prompts

You can also optimize system and instruction prompts:

```python
optimizer = PydanticOptimizer(
    model=User,
    examples=examples,
    evaluate_fn="exact",
    system_prompt="You are a helpful assistant that extracts information.",
    instruction_prompt="Extract the user information from the input text.",
    model_id="gpt-4o"
)

result = optimizer.optimize()

# Access optimized prompts
print(result.optimized_system_prompt)
print(result.optimized_instruction_prompt)
print(result.optimized_descriptions)
```

## Built-in Evaluation Options

Instead of writing a custom evaluation function, you can use built-in options:

- `"exact"`: Exact matching between extracted and expected values
- `"levenshtein"`: Fuzzy matching using Levenshtein distance

```python
optimizer = PydanticOptimizer(
    model=User,
    examples=examples,
    evaluate_fn="exact",  # or "levenshtein"
    model_id="gpt-4o"
)
```

## Examples

See the [examples directory](examples/) for complete working examples:

- **[Text extraction](examples/text_example.py)**: Extract structured information from veterinary EHR text
- **[Image classification](examples/image_example.py)**: Classify handwritten digits from MNIST images
- **[Sentiment analysis](examples/imdb_example.py)**: Classify movie review sentiment

## API Reference

### `PydanticOptimizer`

Main optimizer class.

**Parameters:**

- `model` (type[BaseModel]): The Pydantic model class to optimize
- `examples` (list[Example]): List of examples for optimization
- `evaluate_fn` (Callable | str | None): Evaluation function or built-in option ("exact", "levenshtein"). If None, uses default evaluation.
- `system_prompt` (str | None): Optional initial system prompt to optimize
- `instruction_prompt` (str | None): Optional initial instruction prompt to optimize
- `model_id` (str): LLM model ID (default: "gpt-4o")
- `api_key` (str | None): API key (default: from OPENAI_API_KEY env var)
- `verbose` (bool): Print progress (default: False)
- `optimizer_type` (str): Optimizer type (default: "miprov2zeroshot")
- `num_threads` (int): Number of optimization threads (default: 4)

**Returns:**

- `OptimizationResult`: Contains optimized descriptions, prompts, and metrics

### `Example`

Example data for optimization.

**Parameters:**

- `expected_output` (dict | BaseModel): Expected output as a Pydantic model instance or dict
- `text` (str | None): Plain text input
- `image_path` (str | Path | None): Path to an image file
- `image_base64` (str | None): Base64-encoded image string
- `pdf_path` (str | Path | None): Path to a PDF file
- `pdf_dpi` (int): DPI for PDF conversion (default: 300)

**Example:**

```python
# Text input
Example(
    text="John Doe, 30 years old",
    expected_output=User(name="John Doe", age=30)
)

# Image input
Example(
    image_path="document.png",
    expected_output=User(name="John Doe", age=30)
)

# PDF input
Example(
    pdf_path="document.pdf",
    expected_output=User(name="John Doe", age=30)
)

# Combined text and image
Example(
    text="Extract information from this document",
    image_path="document.png",
    expected_output=User(name="John Doe", age=30)
)
```

### `apply_optimized_descriptions(model, optimized_descriptions)`

Create a JSON schema with optimized field descriptions for use with OpenAI structured outputs or other systems.

**Parameters:**

- `model` (type[BaseModel]): The original Pydantic model class
- `optimized_descriptions` (dict[str, str]): Dictionary mapping field paths to optimized descriptions

**Returns:**

- `dict`: JSON schema dictionary with optimized descriptions

**Example:**

```python
optimized_schema = apply_optimized_descriptions(Invoice, result.optimized_descriptions)

# Use with OpenAI
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract invoice data..."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": Invoice.__name__,
            "schema": optimized_schema,
            "strict": True
        }
    }
)
```

## License

Apache 2.0

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
