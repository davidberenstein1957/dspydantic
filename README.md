# 🚀 DSPydantic: Auto-Optimize Your Pydantic Models with DSPy

Automatically optimize Pydantic model field descriptions and prompts using DSPy. Get better structured data extraction from LLMs with less manual tuning.

DSPydantic automatically optimizes your Pydantic model field descriptions and prompts using DSPy, so you can extract better structured data from LLMs with zero manual tuning.

## ✨ What It Does

Instead of spending hours crafting the perfect field descriptions for your Pydantic models, DSPydantic uses DSPy's optimization algorithms to automatically find the best descriptions based on your examples. Just provide a few examples, and watch your extraction accuracy improve.

## 🎯 Quick Start

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import PydanticOptimizer, Example, create_optimized_model

# 1. Define your model (any Pydantic model works)
class PatientRecord(BaseModel):
    patient_name: str = Field(description="Patient full name")
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="Urgency level of the case"
    )
    diagnosis: str = Field(description="Primary diagnosis")

# 2. Provide examples (just input text + expected output)
examples = [
    Example(
        text="Patient: Sarah Johnson, age 45. Presenting with hypertension.",
        expected_output=PatientRecord(
            patient_name="Sarah Johnson",
            urgency="medium",
            diagnosis="hypertension"
        )
    ),
    Example(
        text="45-year-old Sarah Johnson seen for HTN.",
        expected_output=PatientRecord(
            patient_name="Sarah Johnson",
            urgency="medium",
            diagnosis="HTN"
        )
    ),
]

# 3. Optimize and use
optimizer = PydanticOptimizer(
    model=PatientRecord,
    examples=examples,
    model_id="gpt-4o"
)
result = optimizer.optimize() 

OptimizedPatientRecord = create_optimized_model(
    PatientRecord,
    result.optimized_descriptions
)
# Use OptimizedPatientRecord just like your original model, but with better accuracy!
```

**That's it!** Your model now has optimized descriptions that extract data more accurately.

## 📦 Installation

```bash
pip install dspydantic
```

Or with `uv`:

```bash
uv pip install dspydantic
```

## 🌟 Key Features

- **Auto-optimization**: Finds best field descriptions automatically—20-40% accuracy improvement
- **Simple input**: Just examples (text/images/PDFs) + your Pydantic model
- **Better output**: Optimized model ready to use with improved accuracy
- **Template prompts**: Dynamic prompts with `{placeholders}` for context-aware extraction
- **Enum & Literal support**: Optimize classification models—often 70% → 90%+ accuracy
- **Multiple formats**: Text, images, PDFs—works with any input type
- **Smart defaults**: Auto-selects best optimizer, no configuration needed

## 📚 Examples

Check out the [examples directory](examples/) for complete working examples:

- **[Veterinary EHR extraction](examples/text_example.py)**: Extract diseases, ICD-11 labels, and anonymized entities from clinical narratives—real-world medical data extraction
- **[Image classification](examples/image_example.py)**: Classify MNIST handwritten digits using `Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`—demonstrates vision capabilities and Literal type optimization
- **[Text classification](examples/imdb_example.py)**: Classify IMDB movie review sentiment with `Literal["positive", "negative"]` and template prompts—shows dynamic prompt formatting with `{review}` placeholders
- **[Human-in-the-loop](examples/hitl_example.py)**: Interactive evaluation with GUI—get human feedback during optimization

## Basic Usage

### 1. Define Your Pydantic Model

```python
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice ID")
    total_amount: float = Field(description="Total amount")
    date: str = Field(description="Invoice date")
    vendor_name: str = Field(description="Vendor or supplier name")
    line_items: list[str] = Field(description="List of purchased items")
```

### 2. Create Examples

**Simple input format**—just text + expected output:

```python
from dspydantic import Example

# Plain text input
examples = [
    Example(
        text="Invoice #INV-2024-001 from Acme Corp. Total: $1,200.00",
        expected_output=Invoice(
            invoice_number="INV-2024-001",
            total_amount=1200.00,
            vendor_name="Acme Corp"
        )
    ),
]

# Or use dictionaries for template prompts (see Template Usage section)
# Or use images: Example(image_path="invoice.png", expected_output=...)
# Or use PDFs: Example(pdf_path="invoice.pdf", expected_output=...)
```

### 3. Optimize

```python
from dspydantic import PydanticOptimizer

optimizer = PydanticOptimizer(
    model=Invoice,
    examples=examples,
    model_id="gpt-4o"
)

result = optimizer.optimize()  # Returns optimized descriptions

# Access optimized results
result.optimized_descriptions        # dict[str, str] - optimized field descriptions
result.optimized_system_prompt      # str | None - optimized system prompt
result.optimized_instruction_prompt # str | None - optimized instruction prompt
```

**Template Formatting**: When using `text` as a dictionary, instruction prompt templates with placeholders like `{key}` are automatically formatted with values from each example's text dict. This allows you to create dynamic, example-specific prompts. See the [Template Usage](#template-usage-with-dynamic-prompts) section for a complete example.

### 4. Use Your Optimized Model

**Simple output**—just use the optimized model like your original:

```python
from dspydantic import create_optimized_model
from openai import OpenAI

# Create optimized model (drop-in replacement)
OptimizedInvoice = create_optimized_model(
    Invoice,
    result.optimized_descriptions
)

# Use with OpenAI structured outputs
client = OpenAI()
messages = []

# Add optimized system prompt if available
if result.optimized_system_prompt:
    messages.append({
        "role": "system",
        "content": result.optimized_system_prompt
    })

# Prepare user content with optimized instruction prompt
user_content = (
    "Invoice #INV-2024-003 from Widget Co. dated March 10, 2024. "
    "Items: Widgets (100x $5), Gadgets (50x $10). Total: $1,000.00"
)
if result.optimized_instruction_prompt:
    user_content = f"{result.optimized_instruction_prompt}\n\n{user_content}"

messages.append({
    "role": "user",
    "content": user_content
})

# Call OpenAI API with optimized model
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format=OptimizedInvoice
)

# Parse response using the optimized model
invoice = OptimizedInvoice.model_validate_json(
    response.choices[0].message.content
)
```

**That's it!** Your optimized model extracts data more accurately with zero code changes.

## Working with Images

**Easy input**: Just provide image paths + expected output:

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example, PydanticOptimizer, create_optimized_model

class DigitClassification(BaseModel):
    digit: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = Field(description="Digit 0-9")

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
    model_id="gpt-4o"
)
result = optimizer.optimize()  # Gain: 75% → 94% accuracy

OptimizedDigit = create_optimized_model(
    DigitClassification,
    result.optimized_descriptions
)
```

## Working with PDFs

**Easy input**: Just provide PDF paths + expected output:

```python
examples = [
    Example(
        pdf_path="invoice_001.pdf",
        expected_output=Invoice(
            invoice_number="INV-2024-001",
            total_amount=1234.56
        )
    ),
]
```

## Template Usage with Dynamic Prompts

Use template prompts with placeholders that are automatically filled from example data dictionaries. This is perfect for creating dynamic, context-aware prompts:

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example, PydanticOptimizer, create_optimized_model

class CustomerFeedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment of the review"
    )
    rating: int = Field(description="Rating 1-5")

# Input: dictionary with placeholders
examples = [
    Example(
        text={
            "review": "Great product!",
            "customer": "John",
            "product": "Mouse"
        },
        expected_output=CustomerFeedback(
            sentiment="positive",
            rating=5
        )
    ),
]

# Template prompt with {placeholders} - automatically filled from dict keys
optimizer = PydanticOptimizer(
    model=CustomerFeedback,
    examples=examples,
    instruction_prompt="Analyze review from {customer} about {product}: {review}",
    model_id="gpt-4o"
)

result = optimizer.optimize()

# The optimizer will automatically format the prompt for each example:
# Example 1: "Analyze review from John about Mouse: Great product!"
```

**Output**: Each example gets a customized prompt automatically—no manual formatting needed!

## Working with Enums and Literals

Use `Literal` or `Enum` in your model—works automatically and is used to optimize the extraction process.

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example, PydanticOptimizer, create_optimized_model

class DocumentClassification(BaseModel):
    doc_type: Literal["invoice", "receipt", "contract"] = Field(description="Document type")
    priority: Literal["low", "medium", "high"] = Field(description="Priority")

examples = [
    Example(
        text="Invoice #12345 from Acme Corp. Total: $1,234.56",
        expected_output=DocumentClassification(
            doc_type="invoice",
            priority="high"
        )
    ),
]

optimizer = PydanticOptimizer(
    model=DocumentClassification,
    examples=examples,
    instruction_prompt=(
        "Classify the following document and extract "
        "its type, priority, and language."
    ),
    evaluate_fn="exact",
    model_id="gpt-4o"
)

result = optimizer.optimize()

OptimizedDocumentClassification = create_optimized_model(
    DocumentClassification,
    result.optimized_descriptions
)
```

**Output**: Optimized descriptions help distinguish between similar categories automatically!

## Nested Models

Nested models work automatically—no special handling needed.

```python
class Address(BaseModel):
    street: str = Field(description="Street")
    city: str = Field(description="City")

class Customer(BaseModel):
    name: str = Field(description="Name")
    address: Address = Field(description="Address")

examples = [
    Example(
        text="Jane Smith, 456 Oak Ave, San Francisco",
        expected_output=Customer(
            name="Jane Smith",
            address=Address(
                street="456 Oak Ave",
                city="San Francisco"
            )
        )
    ),
]

# Field paths automatically handled: "name", "address.street", "address.city"
```

## Custom Evaluation

Provide your own evaluation function if needed.

```python
def evaluate(
    example,
    optimized_descriptions,
    system_prompt,
    instruction_prompt
) -> float:
    # Your custom logic - returns score 0.0 to 1.0
    return 0.85

optimizer = PydanticOptimizer(
    model=Customer,
    examples=examples,
    evaluate_fn=evaluate,  # Use custom evaluation
    model_id="gpt-4o"
)
```

## Evaluation Without Expected Output (LLM Judge)

When you don't have ground truth expected outputs, you can use an LLM as a judge to evaluate the quality of extracted data. This is useful when:

- You have unlabeled data—optimize on real-world examples without manual labeling
- You want to evaluate based on quality rather than exact matching—useful for subjective or nuanced extractions
- You need more nuanced evaluation criteria—e.g., "is this a reasonable extraction?" rather than exact match

### Using Default LLM Judge

When `expected_output` is `None`, the optimizer automatically uses the same LLM as a judge:

```python
examples = [
    Example(
        text=(
            "Patient record: John Doe, age 30, contact: john@example.com, "
            "presenting symptoms: persistent cough and fatigue"
        ),
        expected_output=None  # No ground truth, uses LLM judge
    ),
    Example(
        text=(
            "Medical note: Jane Smith, 25 years old, "
            "email jane.smith@email.com, chief complaint: headache and dizziness"
        ),
        expected_output=None
    ),
]

optimizer = PydanticOptimizer(
    model=PatientRecord,
    examples=examples,
    model_id="gpt-4o",  # This LLM will be used as judge
    api_key="your-api-key"
)

result = optimizer.optimize()
```

### Using a Separate Judge LLM

You can pass a different `dspy.LM` as `evaluate_fn` to use as a judge:

```python
import dspy

# Create a separate judge LM (e.g., a more powerful model for judging)
judge_lm = dspy.LM(
    "gpt-4o",
    api_key="your-api-key"
)

examples = [
    Example(
        text="Patient: John Doe, age 30, presenting with acute symptoms",
        expected_output=None
    ),
]

optimizer = PydanticOptimizer(
    model=PatientRecord,
    examples=examples,
    evaluate_fn=judge_lm,  # Pass dspy.LM as evaluate_fn
    model_id="gpt-4o",  # This LLM is used for optimization
    api_key="your-api-key"
)

result = optimizer.optimize()
```

**Note**: When `expected_output` is `None`:

- If `evaluate_fn` is a `dspy.LM`, it will be used as the judge
- If `evaluate_fn` is a callable, it will be treated as a judge function (with `extracted_data` parameter)
- If `evaluate_fn` is `None` or a string ("exact", "levenshtein", "exact-hitl", "levenshtein-hitl"), the default LLM judge will be used

## Optimizing Prompts

Provide optional prompts or template prompts with placeholders—they'll be optimized too.

```python
optimizer = PydanticOptimizer(
    model=PatientRecord,
    examples=examples,
    system_prompt="You are a medical assistant.",  # Optional
    instruction_prompt="Extract patient info. Analyze this: {note}.",  # Optional
    model_id="gpt-4o"
)

result = optimizer.optimize()
# Typical results: baseline 68% → optimized 91% accuracy

# Create optimized model with updated descriptions
from dspydantic import create_optimized_model

OptimizedPatientRecord = create_optimized_model(
    PatientRecord,
    result.optimized_descriptions
)

# Access optimized prompts
print(result.optimized_system_prompt)      # Optimized system prompt
print(result.optimized_instruction_prompt)  # Optimized instruction prompt
print(result.optimized_descriptions)        # Optimized field descriptions
```

## Built-in Evaluation Options

Instead of writing a custom evaluation function, you can use built-in options:

- `"exact"`: Exact matching between extracted and expected values
- `"levenshtein"`: Fuzzy matching using Levenshtein distance
- `"exact-hitl"`: Human-in-the-loop exact evaluation (shows GUI popup)
- `"levenshtein-hitl"`: Human-in-the-loop Levenshtein evaluation (shows GUI popup)

```python
optimizer = PydanticOptimizer(
    model=PatientRecord,
    examples=examples,
    evaluate_fn="exact",  # or "levenshtein" for fuzzy matching
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
# Automatically selects best optimizer based on dataset size
optimizer = PydanticOptimizer(
    model=PatientRecord,
    examples=examples,  # Will auto-select based on len(examples)
    model_id="gpt-4o"
)
```

### Manual Optimizer Selection

You can specify an optimizer by passing it as a string (optimizer type name) or as a Teleprompter instance:

```python
# Use a specific optimizer type (as string)
optimizer = PydanticOptimizer(
    model=PatientRecord,
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
    model=PatientRecord,
    examples=examples,
    optimizer=custom_optimizer  # Pass custom optimizer instance
)
```

**Note**: The `optimizer` parameter accepts either a string (optimizer type name) or a Teleprompter instance. If None, it will auto-select based on dataset size.

## Examples

See the [examples directory](examples/) for complete working examples:

- **[Veterinary EHR extraction](examples/text_example.py)**: Extract diseases, ICD-11 labels, and anonymized entities from clinical narratives—real-world medical data extraction
- **[Handwritten digit classification](examples/image_example.py)**: Classify MNIST handwritten digits from images—demonstrates vision capabilities
- **[Movie review sentiment](examples/imdb_example.py)**: Classify IMDB movie review sentiment with template prompts—shows dynamic prompt formatting

## API Reference

### `PydanticOptimizer`

Main optimizer class.

**Parameters:**

- `model` (type[BaseModel]): The Pydantic model class to optimize—your structured data schema
- `examples` (list[Example]): List of examples for optimization—typically 5-20 examples yield good results
- `evaluate_fn` (Callable | dspy.LM | str | None): Evaluation function, built-in option ("exact", "levenshtein", "exact-hitl", "levenshtein-hitl"), or dspy.LM instance.
  - When `expected_output` is provided: Can be a callable `(Example, dict[str, str], str | None, str | None) -> float`,
    a string ("exact" or "levenshtein"), or None (uses default evaluation).
  - When `expected_output` is None: Can be a `dspy.LM` instance (used as judge), a callable judge function
    `(Example, dict[str, Any], dict[str, str], str | None, str | None) -> float`, or None (uses default LLM judge).
- `system_prompt` (str | None): Optional initial system prompt to optimize—helps set context for your domain
- `instruction_prompt` (str | None): Optional initial instruction prompt to optimize—can include template placeholders like `{key}`
- `lm` (dspy.LM | None): Optional DSPy language model instance. If provided, used instead of creating one from model_id/api_key.
- `model_id` (str): LLM model ID (default: "gpt-4o")—use "gpt-4o-mini" for faster/cheaper optimization
- `api_key` (str | None): API key (default: from OPENAI_API_KEY env var)
- `api_base` (str | None): API base URL (for Azure OpenAI or custom endpoints)
- `api_version` (str | None): API version (for Azure OpenAI)
- `num_threads` (int): Number of optimization threads (default: 4)—increase for faster optimization
- `init_temperature` (float): Initial temperature for optimization (default: 1.0)
- `verbose` (bool): Print progress (default: False)—set True to see optimization progress and scores
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

- `expected_output` (dict | BaseModel | None): Expected output as a Pydantic model instance or dict.
  If `None`, evaluation will use an LLM judge or custom evaluation function instead of comparing
  against expected output.
- `text` (str | None): Plain text input
- `image_path` (str | Path | None): Path to an image file
- `image_base64` (str | None): Base64-encoded image string
- `pdf_path` (str | Path | None): Path to a PDF file
- `pdf_dpi` (int): DPI for PDF conversion (default: 300)

**Example:**

```python
# Text input
Example(
    text="Patient: John Doe, age 30, diagnosed with hypertension",
    expected_output=PatientRecord(
        patient_name="John Doe",
        age=30,
        diagnosis="hypertension",
        medications=[]
    )
)

# Image input
Example(
    image_path="medical_form.png",
    expected_output=PatientRecord(
        patient_name="Jane Smith",
        age=45,
        diagnosis="diabetes",
        medications=["Metformin"]
    )
)

# PDF input
Example(
    pdf_path="patient_record.pdf",
    expected_output=PatientRecord(
        patient_name="Bob Johnson",
        age=52,
        diagnosis="asthma",
        medications=["Albuterol"]
    )
)

# Combined text and image
Example(
    text="Extract patient information from this medical form",
    image_path="medical_form.png",
    expected_output=PatientRecord(
        patient_name="Sarah Williams",
        age=38,
        diagnosis="migraine",
        medications=["Ibuprofen"]
    )
)

# Without expected_output (uses LLM judge for evaluation)
Example(
    text="Patient: John Doe, age 30, presenting with chest pain",
    expected_output=None
)
```

### `create_optimized_model(model, optimized_descriptions)`

**Recommended:** Create a new Pydantic model class with optimized descriptions.

**Parameters:**

- `model` (type[BaseModel]): Your original Pydantic model
- `optimized_descriptions` (dict[str, str]): From `result.optimized_descriptions`

**Returns:**

- `type[BaseModel]`: New model class with optimized descriptions in Field definitions

**Example:**

```python
from dspydantic import create_optimized_model

# Create optimized model with improved field descriptions
OptimizedInvoice = create_optimized_model(
    Invoice,
    result.optimized_descriptions
)

# Use with OpenAI structured outputs
# The optimized descriptions improve extraction accuracy
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format=OptimizedInvoice
)
```

### `apply_optimized_descriptions(model, optimized_descriptions)`

Get optimized JSON schema without creating a new model class. Useful for one-off schema generation.

**Parameters:**

- `model` (type[BaseModel]): Your original Pydantic model
- `optimized_descriptions` (dict[str, str]): From `result.optimized_descriptions`

**Returns:**

- `dict`: JSON schema dictionary with optimized descriptions

**Example:**

```python
from dspydantic import apply_optimized_descriptions

# Get optimized schema directly without creating a new model class
optimized_schema = apply_optimized_descriptions(
    Invoice,
    result.optimized_descriptions
)

# Use with OpenAI structured outputs
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "schema": optimized_schema
        }
    }
)
```

## License

Apache 2.0

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
