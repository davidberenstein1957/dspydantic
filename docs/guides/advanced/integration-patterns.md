# Integration Patterns

Common patterns for integrating DSPydantic into your applications.

---

## FastAPI Web Service

Expose extraction as an API endpoint.

### Basic Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dspydantic import Prompter

app = FastAPI()

class Invoice(BaseModel):
    vendor: str = Field(description="Vendor name")
    total: str = Field(description="Total amount")
    due_date: str = Field(description="Payment due date")

prompter = Prompter.load(
    "./invoice_prompter",
    model=Invoice,
    model_id="openai/gpt-4o-mini"
)

class ExtractionRequest(BaseModel):
    text: str

class ExtractionResponse(BaseModel):
    data: Invoice
    confidence: float

@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(request: ExtractionRequest):
    result = prompter.predict_with_confidence(request.text)
    return ExtractionResponse(data=result.data, confidence=result.confidence)
```

### With Validation

```python
@app.post("/extract")
async def extract_invoice(request: ExtractionRequest):
    result = prompter.predict_with_confidence(request.text)
    
    if result.confidence < 0.7:
        raise HTTPException(
            status_code=422,
            detail=f"Low confidence extraction: {result.confidence:.0%}"
        )
    
    return {"data": result.data.model_dump(), "confidence": result.confidence}
```

### Async Batch Endpoint

```python
class BatchRequest(BaseModel):
    texts: list[str]

@app.post("/extract/batch")
async def extract_batch(request: BatchRequest):
    results = await prompter.apredict_batch(request.texts, max_workers=4)
    return {"results": [r.model_dump() for r in results]}
```

---

## Background Processing

Process documents asynchronously with task queues.

### Celery Worker

```python
from celery import Celery
from dspydantic import Prompter

app = Celery("extraction", broker="redis://localhost:6379")

@app.task
def extract_document(text: str, prompter_path: str, model_id: str):
    from myapp.models import Invoice
    
    prompter = Prompter.load(prompter_path, model=Invoice, model_id=model_id)
    result = prompter.predict_with_confidence(text)
    
    return {
        "data": result.data.model_dump(),
        "confidence": result.confidence
    }

# Usage
result = extract_document.delay(
    text="Invoice from...",
    prompter_path="./invoice_prompter",
    model_id="openai/gpt-4o-mini"
)
```

### AsyncIO Queue

```python
import asyncio
from collections.abc import AsyncIterator

async def process_queue(
    queue: asyncio.Queue,
    prompter: Prompter,
    batch_size: int = 10
) -> AsyncIterator[dict]:
    """Process items from queue in batches."""
    batch = []
    
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
            batch.append(item)
            
            if len(batch) >= batch_size:
                results = await prompter.apredict_batch(batch, max_workers=4)
                for item, result in zip(batch, results):
                    yield {"input": item, "output": result.model_dump()}
                batch = []
                
        except asyncio.TimeoutError:
            if batch:
                results = await prompter.apredict_batch(batch, max_workers=4)
                for item, result in zip(batch, results):
                    yield {"input": item, "output": result.model_dump()}
                batch = []
```

---

## Database Pipeline

Store extractions in a database.

### SQLAlchemy Integration

```python
from sqlalchemy import create_engine, Column, String, Float, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from dspydantic import Prompter

Base = declarative_base()

class ExtractedDocument(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    source_text = Column(String)
    extracted_data = Column(JSON)
    confidence = Column(Float)

engine = create_engine("sqlite:///extractions.db")
Session = sessionmaker(bind=engine)

def process_and_store(texts: list[str], prompter: Prompter):
    """Extract from texts and store in database."""
    session = Session()
    
    for text in texts:
        result = prompter.predict_with_confidence(text)
        
        doc = ExtractedDocument(
            id=str(hash(text)),
            source_text=text,
            extracted_data=result.data.model_dump(),
            confidence=result.confidence
        )
        session.add(doc)
    
    session.commit()
```

### With Confidence Filtering

```python
def process_with_review(texts: list[str], prompter: Prompter, threshold: float = 0.85):
    """Route low-confidence extractions to review queue."""
    session = Session()
    review_queue = []
    
    for text in texts:
        result = prompter.predict_with_confidence(text)
        
        if result.confidence >= threshold:
            doc = ExtractedDocument(
                id=str(hash(text)),
                source_text=text,
                extracted_data=result.data.model_dump(),
                confidence=result.confidence
            )
            session.add(doc)
        else:
            review_queue.append({
                "text": text,
                "extraction": result.data.model_dump(),
                "confidence": result.confidence
            })
    
    session.commit()
    return review_queue
```

---

## File Processing Pipeline

Process documents from a directory.

### Batch File Processing

```python
from pathlib import Path
import json

def process_directory(
    input_dir: str,
    output_dir: str,
    prompter: Prompter,
    pattern: str = "*.txt"
):
    """Process all matching files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    files = list(input_path.glob(pattern))
    texts = [f.read_text() for f in files]
    
    results = prompter.predict_batch(texts, max_workers=4)
    
    for file, result in zip(files, results):
        output_file = output_path / f"{file.stem}.json"
        output_file.write_text(json.dumps(result.model_dump(), indent=2))
```

### PDF Processing

```python
async def process_pdfs(pdf_dir: str, prompter: Prompter):
    """Process all PDFs in a directory."""
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    results = []
    for pdf in pdf_files:
        result = await prompter.apredict(pdf_path=str(pdf))
        results.append({
            "file": pdf.name,
            "extraction": result.model_dump()
        })
    
    return results
```

---

## Monitoring and Logging

Track extraction performance.

### Basic Logging

```python
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_with_logging(text: str, prompter: Prompter):
    """Extract with timing and logging."""
    start = time.time()
    
    result = prompter.predict_with_confidence(text)
    
    elapsed = time.time() - start
    
    logger.info(
        "Extraction complete",
        extra={
            "confidence": result.confidence,
            "elapsed_seconds": elapsed,
            "field_count": len(result.data.model_dump())
        }
    )
    
    return result
```

### Metrics Collection

```python
from dataclasses import dataclass, field
from typing import ClassVar

@dataclass
class ExtractionMetrics:
    total_extractions: int = 0
    total_time_seconds: float = 0.0
    confidence_sum: float = 0.0
    low_confidence_count: int = 0
    
    _instance: ClassVar["ExtractionMetrics | None"] = None
    
    @classmethod
    def get(cls) -> "ExtractionMetrics":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def record(self, confidence: float, elapsed: float):
        self.total_extractions += 1
        self.total_time_seconds += elapsed
        self.confidence_sum += confidence
        if confidence < 0.8:
            self.low_confidence_count += 1
    
    @property
    def average_confidence(self) -> float:
        if self.total_extractions == 0:
            return 0.0
        return self.confidence_sum / self.total_extractions
    
    @property
    def average_time(self) -> float:
        if self.total_extractions == 0:
            return 0.0
        return self.total_time_seconds / self.total_extractions

def extract_with_metrics(text: str, prompter: Prompter):
    """Extract and record metrics."""
    start = time.time()
    result = prompter.predict_with_confidence(text)
    elapsed = time.time() - start
    
    ExtractionMetrics.get().record(result.confidence, elapsed)
    
    return result
```

---

## Error Handling

Handle extraction failures gracefully.

### Retry Pattern

```python
import time
from typing import TypeVar

T = TypeVar("T")

def extract_with_retry(
    text: str,
    prompter: Prompter,
    max_retries: int = 3,
    backoff_factor: float = 2.0
):
    """Extract with exponential backoff retry."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return prompter.run(text)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
    
    raise last_error
```

### Fallback Pattern

```python
def extract_with_fallback(text: str, primary: Prompter, fallback: Prompter):
    """Try primary prompter, fall back to secondary on failure."""
    try:
        result = primary.predict_with_confidence(text)
        if result.confidence >= 0.7:
            return result.data
    except Exception:
        pass
    
    return fallback.run(text)
```

---

## Development Workflow

Optimize iteration speed during development.

### Caching for Development

```python
# Enable caching during development
prompter = Prompter(
    model=Invoice,
    model_id="openai/gpt-4o-mini",
    cache=True
)

# Same input returns cached result (no API call)
result1 = prompter.run("Invoice from Acme...")
result2 = prompter.run("Invoice from Acme...")  # Cached
```

### A/B Testing Prompters

```python
import random

def ab_test_extraction(text: str, prompter_a: Prompter, prompter_b: Prompter):
    """Run A/B test between two prompters."""
    if random.random() < 0.5:
        result = prompter_a.predict_with_confidence(text)
        variant = "A"
    else:
        result = prompter_b.predict_with_confidence(text)
        variant = "B"
    
    return {
        "variant": variant,
        "data": result.data.model_dump(),
        "confidence": result.confidence
    }
```

---

## See Also

- [Save and Load](save-load.md) - Production deployment
- [Configure Models](configure-models.md) - Model configuration
- [Save and Load](save-load.md) - Production deployment
