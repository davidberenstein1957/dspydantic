# dspydantic

Optimize information extraction prompts and Pydantic field descriptions using DSPy.

## Overview

`dspydantic` is a library that uses [DSPy](https://github.com/stanfordnlp/dspy) to automatically optimize field descriptions in [Pydantic](https://docs.pydantic.dev/) models. By providing example data and an evaluation function, the library iteratively improves field descriptions to achieve better structured output quality from LLMs.

## Why dspydantic?

When building LLM-powered applications that extract structured data, getting the right field descriptions and prompts is crucial for accuracy. Manually crafting and iterating on these descriptions is time-consuming and often suboptimal. `dspydantic` solves this by:

- **Automatically optimizing field descriptions**: Instead of manually tweaking descriptions, provide examples and let DSPy find the optimal descriptions that maximize extraction accuracy
- **Optimizing system and instruction prompts**: Beyond field descriptions, optimize the entire prompt structure for better results
- **Data-driven improvements**: Uses your actual data and evaluation metrics to iteratively improve, rather than relying on intuition
- **Works with any input format**: Supports plain text, images (including PDFs converted to images), and combinations thereof

### Use Cases

`dspydantic` is particularly useful for:

- **Document extraction**: Extract structured data from invoices, forms, medical records, or any document format
- **Image analysis**: Extract information from images, diagrams, or scanned documents
- **Text classification**: Optimize models for sentiment analysis, categorization, or any text classification task
- **Multi-modal extraction**: Combine text and images for complex extraction scenarios

See the [examples directory](examples/) for complete working examples:

- **[Text extraction](examples/text_example.py)**: Extract structured information from veterinary EHR text (PetEVAL dataset)
- **[Image classification](examples/image_example.py)**: Classify handwritten digits from MNIST images
- **[Sentiment analysis](examples/imdb_example.py)**: Classify movie review sentiment from IMDB dataset

## Features

- 🔄 **Automatic Optimization**: Uses DSPy to optimize Pydantic field descriptions
- 📊 **Custom Evaluation**: Define your own evaluation function to measure quality
- 🎯 **Multiple Optimizers**: Support for MIPROv2, GEPA, BootstrapFewShot, and more
- 🔧 **Easy Integration**: Simple API that works with any Pydantic model
- 📝 **Recursive Support**: Handles nested models and arrays of objects

## Installation

```bash
pip install dspydantic
```

Or using `uv`:

```bash
uv pip install dspydantic
```

## Quick Start

### Text Input Example

```python
from pydantic import BaseModel, Field
from dspydantic import PydanticOptimizer, Example

# Define your Pydantic model
class User(BaseModel):
    name: str = Field(description="User name")
    age: int = Field(description="User age")
    email: str = Field(description="Email address")

# Prepare examples with text input
examples = [
    Example(
        input_data={"text": "John Doe, 30 years old, john@example.com"},
        expected_output={"name": "John Doe", "age": 30, "email": "john@example.com"}
    ),
    Example(
        input_data={"text": "Jane Smith, 25, jane.smith@email.com"},
        expected_output={"name": "Jane Smith", "age": 25, "email": "jane.smith@email.com"}
    ),
]

# Optimize with built-in evaluation (uses "exact" matching)
optimizer = PydanticOptimizer(
    model=User,
    examples=examples,
    evaluate_fn="exact",  # Built-in exact matching evaluation
    model_id="gpt-4o",
    api_key="your-api-key",  # Or set OPENAI_API_KEY env var
    verbose=True
)

# Run optimization
result = optimizer.optimize()

# View optimized descriptions
print("Optimized descriptions:")
for field, description in result.optimized_descriptions.items():
    print(f"  {field}: {description}")

# Use optimized descriptions with OpenAI structured outputs
from dspydantic import apply_optimized_descriptions
from openai import OpenAI

optimized_schema = apply_optimized_descriptions(User, result.optimized_descriptions)
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract: Alice Johnson, 28, alice@example.com"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": User.__name__,
            "schema": optimized_schema,
            "strict": True
        }
    }
)
```

### Image Input Example

```python
from pydantic import BaseModel, Field
from dspydantic import PydanticOptimizer, Example, prepare_input_data
from typing import Literal

# Define model for image classification
class DigitClassification(BaseModel):
    digit: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = Field(
        description="The digit shown in the image (0-9)"
    )

# Prepare examples with image input
examples = [
    Example(
        input_data=prepare_input_data(image_path="digit_5.png"),
        expected_output={"digit": 5}
    ),
    Example(
        input_data=prepare_input_data(image_path="digit_3.png"),
        expected_output={"digit": 3}
    ),
]

# Optimize
optimizer = PydanticOptimizer(
    model=DigitClassification,
    examples=examples,
    evaluate_fn="exact",
    model_id="gpt-4o",
    api_key="your-api-key",
    verbose=True
)

result = optimizer.optimize()
```

### PDF Input Example

```python
from pydantic import BaseModel, Field
from dspydantic import PydanticOptimizer, Example, prepare_input_data

# Define model for invoice extraction
class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice number")
    total_amount: float = Field(description="Total amount")
    date: str = Field(description="Invoice date")

# Prepare examples with PDF input
examples = [
    Example(
        input_data=prepare_input_data(pdf_path="invoice_001.pdf"),
        expected_output={
            "invoice_number": "INV-2024-001",
            "total_amount": 1234.56,
            "date": "2024-01-15"
        }
    ),
    Example(
        input_data=prepare_input_data(pdf_path="invoice_002.pdf"),
        expected_output={
            "invoice_number": "INV-2024-002",
            "total_amount": 567.89,
            "date": "2024-01-20"
        }
    ),
]

# Optimize
optimizer = PydanticOptimizer(
    model=Invoice,
    examples=examples,
    evaluate_fn="exact",
    model_id="gpt-4o",
    api_key="your-api-key",
    verbose=True
)

result = optimizer.optimize()
```

### Combined Text and Image Example

```python
from dspydantic import prepare_input_data, Example

# Combine text and image in a single example
examples = [
    Example(
        input_data=prepare_input_data(
            text="Extract information from this receipt",
            image_path="receipt.png"
        ),
        expected_output={"total": 45.99, "merchant": "Coffee Shop"}
    ),
]
```

## Usage

### Basic Example

```python
from pydantic import BaseModel, Field
from dspydantic import PydanticOptimizer, Example, extract_field_descriptions, apply_optimized_descriptions

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice ID")
    total_amount: float = Field(description="Total amount")
    date: str = Field(description="Invoice date")

# Step 1: Inspect current field descriptions (optional)
current_descriptions = extract_field_descriptions(Invoice)
print("Current descriptions:", current_descriptions)
# Output: {
#     "invoice_number": "Invoice ID",
#     "total_amount": "Total amount",
#     "date": "Invoice date"
# }

# Step 2: Prepare examples
examples = [
    Example(
        input_data={"text": "Invoice #INV-2024-001, Total: $1,234.56, Date: 2024-01-15"},
        expected_output={
            "invoice_number": "INV-2024-001",
            "total_amount": 1234.56,
            "date": "2024-01-15"
        }
    ),
    # Add more examples...
]

# Step 3: Optimize field descriptions
optimizer = PydanticOptimizer(
    model=Invoice,
    examples=examples,
    evaluate_fn="exact",  # Use built-in exact matching evaluation
    model_id="gpt-4o"
)
result = optimizer.optimize()

# Step 4: View optimized descriptions
print("\nOptimized descriptions:")
for field_path, description in result.optimized_descriptions.items():
    print(f"  {field_path}: {description}")

# Step 5: Apply optimized descriptions to create a JSON schema
optimized_schema = apply_optimized_descriptions(Invoice, result.optimized_descriptions)

# Step 6: Use with OpenAI structured outputs
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Extract invoice data from: INV-2024-001, $1,234.56, 2024-01-15"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": Invoice.__name__,
            "schema": optimized_schema,
            "strict": True
        }
    }
)

extracted_data = response.choices[0].message.content
print("\nExtracted data:", extracted_data)
```

### Custom Evaluation Function

The evaluation function receives an `Example`, optimized field descriptions, and optimized prompts, and should return a score between 0.0 and 1.0:

```python
def evaluate(
    example: Example,
    optimized_descriptions: dict[str, str],
    optimized_system_prompt: str | None,
    optimized_instruction_prompt: str | None,
) -> float:
    """
    Evaluate how well the optimized prompts and descriptions work.
    
    Args:
        example: The example with input_data and expected_output
        optimized_descriptions: Dictionary of field paths to optimized descriptions
        optimized_system_prompt: Optimized system prompt (None if not provided)
        optimized_instruction_prompt: Optimized instruction prompt (None if not provided)
    
    Returns:
        Score between 0.0 and 1.0
    """
    # Example: Use an LLM to extract data and compare with expected output
    # Use the optimized prompts and descriptions with your LLM
    # This is a simplified example - your actual implementation would
    # call your LLM with the optimized prompts/descriptions and compare results
    
    # For demonstration, return a mock score
    return 0.85
```

### System and Instruction Prompts

You can optimize system prompts and instruction prompts alongside field descriptions:

```python
optimizer = PydanticOptimizer(
    model=User,
    examples=examples,
    evaluate_fn=evaluate,
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

### Custom DSPy Language Model

You can pass any DSPy language model directly instead of using `model_id`:

```python
import dspy
from dspydantic import PydanticOptimizer, Example

# Create a custom DSPy LM with any configuration
custom_lm = dspy.LM(
    "gpt-4o",
    api_key="your-key",
    api_base="https://custom-endpoint.com",  # For custom endpoints
    api_version="2024-01-01",  # For Azure
    # ... any other DSPy LM parameters
)

optimizer = PydanticOptimizer(
    model=User,
    examples=examples,
    evaluate_fn=evaluate,
    lm=custom_lm,  # Pass your custom LM
    verbose=True
)
```

This is useful when you need:

- Custom API endpoints
- Special LM configurations
- Reusing an existing LM instance
- Using DSPy's advanced LM features

### Optimizer Types

Choose from different DSPy optimizers:

- `"miprov2zeroshot"` (default): MIPROv2 configured for 0-shot optimization
- `"miprov2"`: Full MIPROv2 optimization
- `"gepa"`: GEPA optimizer
- `"bootstrapfewshot"`: BootstrapFewShot optimizer
- `"bootstrapfewshotwithrandomsearch"`: BootstrapFewShotWithRandomSearch

```python
optimizer = PydanticOptimizer(
    model=User,
    examples=examples,
    evaluate_fn=evaluate,
    optimizer_type="miprov2",  # Choose optimizer
    num_threads=4,
    verbose=True
)
```

### Nested Models

The library automatically handles nested Pydantic models:

```python
class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")

class User(BaseModel):
    name: str = Field(description="User name")
    address: Address = Field(description="User address")

# Field paths will be: "name", "address.street", "address.city"
```

### Working with Field Descriptions Directly

You can use `extract_field_descriptions` and `apply_optimized_descriptions` independently to inspect and modify field descriptions without running optimization:

```python
from pydantic import BaseModel, Field
from dspydantic import extract_field_descriptions, apply_optimized_descriptions

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price")
    in_stock: bool = Field(description="Availability")

# Extract current descriptions
descriptions = extract_field_descriptions(Product)
print(descriptions)
# {'name': 'Product name', 'price': 'Price', 'in_stock': 'Availability'}

# Manually improve descriptions (or use optimization results)
improved_descriptions = {
    "name": "The full product name as displayed to customers",
    "price": "Price in USD without currency symbol",
    "in_stock": "True if item is currently available for purchase"
}

# Apply improved descriptions to create a schema
optimized_schema = apply_optimized_descriptions(Product, improved_descriptions)

# Use the optimized schema with any LLM that accepts JSON schemas
```

**Use Cases:**

- **Inspect current descriptions**: See what descriptions are currently set in your model
- **Manual refinement**: Manually improve descriptions based on testing or domain knowledge
- **Schema generation**: Create production-ready JSON schemas with optimized descriptions
- **Integration**: Prepare schemas for OpenAI, Anthropic, or other structured output APIs

## API Reference

### `PydanticOptimizer`

Main optimizer class.

**Parameters:**

- `model` (type[BaseModel]): The Pydantic model class to optimize
- `examples` (list[Example]): List of examples for optimization
- `evaluate_fn` (Callable): Function that evaluates quality. Receives (Example, optimized_descriptions, optimized_system_prompt, optimized_instruction_prompt) and returns 0.0-1.0
- `system_prompt` (str | None): Optional initial system prompt to optimize
- `instruction_prompt` (str | None): Optional initial instruction prompt to optimize
- `lm` (dspy.LM | None): Optional DSPy language model instance. If provided, this will be used instead of creating a new one. If None, a new dspy.LM will be created from `model_id`/`api_key`/etc.
- `model_id` (str): LLM model ID (default: "gpt-4o"). Only used if `lm` is None.
- `api_key` (str | None): API key (default: from OPENAI_API_KEY env var). Only used if `lm` is None.
- `api_base` (str | None): API base URL for Azure/custom endpoints. Only used if `lm` is None.
- `api_version` (str | None): API version for Azure. Only used if `lm` is None.
- `num_threads` (int): Number of optimization threads (default: 4)
- `init_temperature` (float): Initial temperature (default: 1.0)
- `verbose` (bool): Print progress (default: False)
- `optimizer_type` (str): Optimizer type (default: "miprov2zeroshot")
- `train_split` (float): Training split ratio (default: 0.8)

**Returns:**

- `OptimizationResult`: Contains optimized descriptions and metrics

### `extract_field_descriptions(model)`

Extract field descriptions from a Pydantic model recursively.

**Parameters:**

- `model` (type[BaseModel]): The Pydantic model class to extract descriptions from

**Returns:**

- `dict[str, str]`: Dictionary mapping field paths to their descriptions. Field paths use dot notation for nested fields (e.g., `"address.street"`).

**Example:**

```python
from pydantic import BaseModel, Field
from dspydantic import extract_field_descriptions

# Simple model
class User(BaseModel):
    name: str = Field(description="User's full name")
    age: int = Field(description="User's age in years")
    email: str = Field(description="User's email address")

descriptions = extract_field_descriptions(User)
# Returns: {
#     "name": "User's full name",
#     "age": "User's age in years",
#     "email": "User's email address"
# }

# Nested model
class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    zip_code: str = Field(description="ZIP code")

class Person(BaseModel):
    name: str = Field(description="Person's name")
    address: Address = Field(description="Home address")
    phone_numbers: list[str] = Field(description="List of phone numbers")

descriptions = extract_field_descriptions(Person)
# Returns: {
#     "name": "Person's name",
#     "address": "Home address",
#     "address.street": "Street address",
#     "address.city": "City name",
#     "address.zip_code": "ZIP code",
#     "phone_numbers": "List of phone numbers"
# }

# Use case: Inspect current descriptions before optimization
current_descriptions = extract_field_descriptions(Invoice)
print("Current field descriptions:")
for field_path, description in current_descriptions.items():
    print(f"  {field_path}: {description}")
```

### `apply_optimized_descriptions(model, optimized_descriptions)`

Create a modified JSON schema with optimized field descriptions applied. This is useful for creating schemas compatible with OpenAI structured outputs, Anthropic, or other systems that accept JSON schemas.

**Parameters:**

- `model` (type[BaseModel]): The original Pydantic model class
- `optimized_descriptions` (dict[str, str]): Dictionary mapping field paths to optimized descriptions

**Returns:**

- `dict[str, Any]`: Modified JSON schema dictionary with optimized descriptions. For OpenAI structured outputs, wrap it as shown in the examples below.

**Example - Basic Usage:**

```python
from pydantic import BaseModel, Field
from dspydantic import apply_optimized_descriptions

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice ID")
    total_amount: float = Field(description="Total amount")
    date: str = Field(description="Invoice date")

# After optimization, you have optimized descriptions
optimized_descriptions = {
    "invoice_number": "The unique alphanumeric identifier found at the top of the invoice",
    "total_amount": "The final amount due including all taxes and fees",
    "date": "The invoice date in YYYY-MM-DD format"
}

# Apply optimized descriptions to create a JSON schema
optimized_schema = apply_optimized_descriptions(Invoice, optimized_descriptions)

# Use with OpenAI structured outputs
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Extract invoice data from: INV-2024-001, $1,234.56, 2024-01-15"}
    ],
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

**Example - Nested Models:**

```python
from pydantic import BaseModel, Field
from dspydantic import apply_optimized_descriptions

class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str = Field(description="State abbreviation")

class Customer(BaseModel):
    name: str = Field(description="Customer name")
    email: str = Field(description="Email address")
    address: Address = Field(description="Billing address")

# Optimized descriptions for nested fields use dot notation
optimized_descriptions = {
    "name": "The customer's full legal name",
    "email": "Primary contact email address",
    "address": "Complete billing address information",
    "address.street": "Street number and name",
    "address.city": "City name (not abbreviated)",
    "address.state": "Two-letter US state code (e.g., CA, NY)"
}

# Apply to create optimized schema
optimized_schema = apply_optimized_descriptions(Customer, optimized_descriptions)

# The schema now has optimized descriptions at all levels
print(optimized_schema["properties"]["address"]["properties"]["street"]["description"])
# Output: "Street number and name"
```

**Example - Complete Workflow:**

```python
from pydantic import BaseModel, Field
from dspydantic import (
    PydanticOptimizer,
    Example,
    extract_field_descriptions,
    apply_optimized_descriptions
)

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Product price")
    category: str = Field(description="Product category")

# Step 1: Extract current descriptions (optional, for inspection)
current_descriptions = extract_field_descriptions(Product)
print("Before optimization:", current_descriptions)

# Step 2: Prepare examples and optimize
examples = [
    Example(
        input_data={"text": "iPhone 15 Pro, $999, Electronics"},
        expected_output={"name": "iPhone 15 Pro", "price": 999.0, "category": "Electronics"}
    ),
    # ... more examples
]

optimizer = PydanticOptimizer(
    model=Product,
    examples=examples,
    model_id="gpt-4o",
    evaluate_fn="exact"
)

result = optimizer.optimize()

# Step 3: Apply optimized descriptions to create a production-ready schema
optimized_schema = apply_optimized_descriptions(Product, result.optimized_descriptions)

# Step 4: Use the optimized schema with your LLM
openai_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": Product.__name__,
        "schema": optimized_schema,
        "strict": True
    }
}

# Now use openai_schema in your API calls
```

**Example - Comparing Before and After:**

```python
from pydantic import BaseModel, Field
from dspydantic import extract_field_descriptions, apply_optimized_descriptions
import json

class Document(BaseModel):
    title: str = Field(description="Document title")
    author: str = Field(description="Author name")
    pages: int = Field(description="Number of pages")

# Get original descriptions
original_descriptions = extract_field_descriptions(Document)
print("Original descriptions:")
for path, desc in original_descriptions.items():
    print(f"  {path}: {desc}")

# After optimization, you have improved descriptions
optimized_descriptions = {
    "title": "The main title of the document, typically found at the top of the first page",
    "author": "The full name of the person or organization who created the document",
    "pages": "The total number of pages in the document as a whole number"
}

# Create schemas for comparison
original_schema = Document.model_json_schema()
optimized_schema = apply_optimized_descriptions(Document, optimized_descriptions)

# Compare field descriptions
print("\nComparison:")
for field_name in original_schema["properties"]:
    original_desc = original_schema["properties"][field_name].get("description", "N/A")
    optimized_desc = optimized_schema["properties"][field_name].get("description", "N/A")
    print(f"\n{field_name}:")
    print(f"  Original:  {original_desc}")
    print(f"  Optimized: {optimized_desc}")
```

## License

Apache 2.0

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
