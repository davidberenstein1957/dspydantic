# DSPydantic

Automatically optimize Pydantic model field descriptions and prompts using DSPy. Get better structured data extraction from LLMs with less manual tuning.

## What is DSPydantic?

When building LLM applications that extract structured data, getting the right field descriptions and prompts is crucial. Instead of manually tweaking descriptions, `dspydantic` uses DSPy to automatically find the best descriptions and prompts based on your examples.

## Quick Start

```python
from pydantic import BaseModel, Field
from dspydantic import PydanticOptimizer, Example, create_optimized_model

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
# Optimizer is auto-selected based on dataset size (< 20 examples uses BootstrapFewShot,
# >= 20 examples uses BootstrapFewShotWithRandomSearch)
# You can also specify: optimizer="miprov2" or pass a custom Teleprompter instance
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

# Create optimized model with updated descriptions
OptimizedUser = create_optimized_model(User, result.optimized_descriptions)

# Use the optimized model directly
user = OptimizedUser(name="John Doe", age=30, email="john@example.com")
print(f"Model schema: {OptimizedUser.model_json_schema()}")
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

You can create a new optimized model class with the optimized descriptions applied directly:

```python
from dspydantic import create_optimized_model
from openai import OpenAI

# Create optimized model class with updated Field descriptions
OptimizedInvoice = create_optimized_model(Invoice, result.optimized_descriptions)

# Use the optimized model directly - it has optimized descriptions in Field definitions
# The optimized model works exactly like the original, but with better descriptions
optimized_schema = OptimizedInvoice.model_json_schema()

# Use with OpenAI structured outputs
# Include optimized system and instruction prompts if they were optimized
client = OpenAI()
messages = []
if result.optimized_system_prompt:
    messages.append({"role": "system", "content": result.optimized_system_prompt})

user_content = "Extract: INV-2024-001, $1,234.56, 2024-01-15"
if result.optimized_instruction_prompt:
    user_content = f"{result.optimized_instruction_prompt}\n\n{user_content}"
messages.append({"role": "user", "content": user_content})

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format=OptimizedInvoice
    
)

# Parse response using the optimized model
invoice = OptimizedInvoice.model_validate_json(response.choices[0].message.content)
```

Alternatively, you can use `apply_optimized_descriptions` to get just the JSON schema without creating a new model class (useful for one-off schema generation).

## Working with Images

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example, PydanticOptimizer, create_optimized_model

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

# Create optimized model
OptimizedDigitClassification = create_optimized_model(
    DigitClassification, result.optimized_descriptions
)

# Use the optimized model
digit = OptimizedDigitClassification(digit=5)
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

# Create optimized model with updated descriptions
from dspydantic import create_optimized_model
OptimizedUser = create_optimized_model(User, result.optimized_descriptions)

# Access optimized prompts
print("Optimized system prompt:", result.optimized_system_prompt)
print("Optimized instruction prompt:", result.optimized_instruction_prompt)
print("Optimized descriptions:", result.optimized_descriptions)

# Use the optimized model and prompts with your LLM
from openai import OpenAI

client = OpenAI()
messages = []
if result.optimized_system_prompt:
    messages.append({"role": "system", "content": result.optimized_system_prompt})

user_content = "John Doe, 123 Main St, New York, 10001"
if result.optimized_instruction_prompt:
    user_content = f"{result.optimized_instruction_prompt}\n\n{user_content}"
messages.append({"role": "user", "content": user_content})

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": OptimizedUser.__name__,
            "schema": OptimizedUser.model_json_schema(),
            "strict": True
        }
    }
)

# Parse response using the optimized model
user = OptimizedUser.model_validate_json(response.choices[0].message.content)
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

## Optimizer Selection

DSPydantic supports multiple DSPy optimizers and automatically selects the best one based on your dataset size, or you can specify one manually.

### Auto-Selection (Default)

If you don't specify an optimizer, DSPydantic will automatically select one based on your dataset size:

- **< 20 examples**: Uses `BootstrapFewShot` (good for small datasets)
- **>= 20 examples**: Uses `BootstrapFewShotWithRandomSearch` (better for larger datasets)

```python
# Auto-selects optimizer based on dataset size
optimizer = PydanticOptimizer(
    model=User,
    examples=examples,  # Will auto-select based on len(examples)
    model_id="gpt-4o"
)
```

### Manual Optimizer Selection

You can specify an optimizer by passing it as a string (optimizer type name) or as a Teleprompter instance:

```python
# Use a specific optimizer type (as string)
optimizer = PydanticOptimizer(
    model=User,
    examples=examples,
    optimizer="miprov2",  # or "gepa", "copro", "simba", etc.
    model_id="gpt-4o"
)
```

Available optimizer types include:

- `"bootstrapfewshot"`: BootstrapFewShot optimizer
- `"bootstrapfewshotwithrandomsearch"`: BootstrapFewShotWithRandomSearch
- `"miprov2"`: MIPROv2 optimizer (best for instruction optimization)
- `"miprov2zeroshot"`: MIPROv2 with zero-shot settings
- `"gepa"`: GEPA optimizer (reflective prompt evolution)
- `"copro"`: COPRO optimizer
- `"simba"`: SIMBA optimizer
- `"knnfewshot"`: KNNFewShot optimizer
- `"labeledfewshot"`: LabeledFewShot optimizer
- And more (all Teleprompter subclasses are automatically supported)

### Custom Optimizer Instance

You can also pass a custom optimizer instance directly:

```python
from dspy.teleprompt import MIPROv2

# Create a custom optimizer with specific settings
custom_optimizer = MIPROv2(
    metric=my_metric_function,
    num_threads=8,
    auto="full",
    max_bootstrapped_demos=8
)

# Use the custom optimizer
optimizer = PydanticOptimizer(
    model=User,
    examples=examples,
    optimizer=custom_optimizer  # Pass custom optimizer instance
)
```

**Note**: The `optimizer` parameter accepts either a string (optimizer type name) or a Teleprompter instance. If None, it will auto-select based on dataset size.

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
- `lm` (dspy.LM | None): Optional DSPy language model instance. If provided, used instead of creating one from model_id/api_key.
- `model_id` (str): LLM model ID (default: "gpt-4o")
- `api_key` (str | None): API key (default: from OPENAI_API_KEY env var)
- `api_base` (str | None): API base URL (for Azure OpenAI or custom endpoints)
- `api_version` (str | None): API version (for Azure OpenAI)
- `num_threads` (int): Number of optimization threads (default: 4)
- `init_temperature` (float): Initial temperature for optimization (default: 1.0)
- `verbose` (bool): Print progress (default: False)
- `optimizer` (str | Teleprompter | None): Optimizer specification. Can be:
  - A string (optimizer type name): e.g., "miprov2", "gepa", "bootstrapfewshot", etc.
    If None, optimizer will be auto-selected based on dataset size.
  - A Teleprompter instance: Custom optimizer instance to use directly.
  Valid optimizer type strings include: "bootstrapfewshot", "bootstrapfewshotwithrandomsearch",
  "miprov2", "gepa", "copro", "simba", etc. (all Teleprompter subclasses are supported)
- `train_split` (float): Fraction of examples to use for training (rest for validation) (default: 0.8)
- `optimizer_kwargs` (dict[str, Any] | None): Optional dictionary of additional keyword arguments
  to pass to the optimizer constructor. Only used if `optimizer` is a string or None.

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

### `create_optimized_model(model, optimized_descriptions)`

Create a new Pydantic model class with optimized field descriptions applied directly to Field definitions. This is the recommended way to use optimized descriptions.

**Parameters:**

- `model` (type[BaseModel]): The original Pydantic model class
- `optimized_descriptions` (dict[str, str]): Dictionary mapping field paths to optimized descriptions

**Returns:**

- `type[BaseModel]`: A new Pydantic model class with optimized descriptions in Field definitions

**Example:**

```python
from dspydantic import create_optimized_model

# Create optimized model class
OptimizedInvoice = create_optimized_model(Invoice, result.optimized_descriptions)

# Use the optimized model directly - it works exactly like the original
# but with optimized descriptions embedded in Field definitions
invoice = OptimizedInvoice(
    invoice_number="INV-2024-001",
    total_amount=1234.56,
    date="2024-01-15"
)

# Get JSON schema with optimized descriptions
optimized_schema = OptimizedInvoice.model_json_schema()

# Use with OpenAI structured outputs
# Include optimized prompts if available
messages = []
if result.optimized_system_prompt:
    messages.append({"role": "system", "content": result.optimized_system_prompt})

user_content = "Extract invoice data..."
if result.optimized_instruction_prompt:
    user_content = f"{result.optimized_instruction_prompt}\n\n{user_content}"
messages.append({"role": "user", "content": user_content})

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": OptimizedInvoice.__name__,
            "schema": optimized_schema,
            "strict": True
        }
    }
)
```

### `apply_optimized_descriptions(model, optimized_descriptions)`

Create a JSON schema dictionary with optimized field descriptions. Useful for one-off schema generation without creating a new model class.

**Parameters:**

- `model` (type[BaseModel]): The original Pydantic model class
- `optimized_descriptions` (dict[str, str]): Dictionary mapping field paths to optimized descriptions

**Returns:**

- `dict`: JSON schema dictionary with optimized descriptions

**Example:**

```python
from dspydantic import apply_optimized_descriptions

# Get optimized schema without creating a new model class
optimized_schema = apply_optimized_descriptions(Invoice, result.optimized_descriptions)

# Use with OpenAI
# Include optimized prompts if available
messages = []
if result.optimized_system_prompt:
    messages.append({"role": "system", "content": result.optimized_system_prompt})

user_content = "Extract invoice data..."
if result.optimized_instruction_prompt:
    user_content = f"{result.optimized_instruction_prompt}\n\n{user_content}"
messages.append({"role": "user", "content": user_content})

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
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
