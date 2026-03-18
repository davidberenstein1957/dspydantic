# Integrate with Applications

Real-world patterns for using DSPydantic in production applications.

## Database Pipeline

Store extractions in a database with confidence-based routing:

```python
from sqlalchemy import create_engine, Column, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ExtractedDocument(Base):
    __tablename__ = "extracted_documents"
    id = Column(String, primary_key=True)
    content = Column(String)
    confidence = Column(Float)
    status = Column(String)  # "approved" or "review"

# Extract and store
def process_document(text):
    result = prompter.run(text)

    session = Session()
    doc = ExtractedDocument(
        id=generate_id(),
        content=str(result.dict()),
        confidence=result.confidence if hasattr(result, 'confidence') else 1.0,
        status="review" if result.confidence < 0.9 else "approved"
    )
    session.add(doc)
    session.commit()
```

---

## Batch File Processing

Process directories of files:

```python
from pathlib import Path
from dspydantic import Prompter

def process_directory(directory_path):
    results = []

    for file_path in Path(directory_path).glob("*.txt"):
        with open(file_path) as f:
            text = f.read()

        result = prompter.run(text)
        results.append({
            "file": file_path.name,
            "data": result.dict()
        })

    return results

# For PDFs:
def process_pdfs(directory_path):
    results = []

    for pdf_path in Path(directory_path).glob("*.pdf"):
        result = prompter.run(pdf_path=str(pdf_path))
        results.append(result)

    return results
```

---

## Background Job Processing

Use async queues for long-running extraction:

```python
import asyncio
from queue import Queue

class ExtractionQueue:
    def __init__(self, prompter):
        self.prompter = prompter
        self.queue = Queue()

    async def process_batch(self):
        while not self.queue.empty():
            text = self.queue.get()
            result = self.prompter.run(text)
            await self.store_result(result)
            self.queue.task_done()

    async def store_result(self, result):
        # Store to database or file
        pass

# Usage
queue = ExtractionQueue(prompter)
await queue.process_batch()
```

---

## Error Handling

Implement retry and fallback patterns:

```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3, backoff_factor=2):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    time.sleep(wait_time)
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
async def extract_with_retry(text):
    return prompter.run(text)

# Fallback pattern
try:
    result = primary_prompter.run(text)
except Exception:
    result = fallback_prompter.run(text)
```

---

## Monitoring

Track extraction metrics:

```python
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractionMetrics:
    total_extractions: int = 0
    failed_extractions: int = 0
    avg_confidence: float = 0.0
    total_api_calls: int = 0

def extract_with_monitoring(text):
    try:
        result = prompter.run(text)
        metrics.total_extractions += 1
        logger.info(f"Extraction succeeded: {result}")
        return result
    except Exception as e:
        metrics.failed_extractions += 1
        logger.error(f"Extraction failed: {e}")
        raise

metrics = ExtractionMetrics()
```

---

## Tips

- Use async for I/O-heavy operations
- Implement monitoring and logging
- Use retry patterns for API failures
- Store confidence scores for quality tracking
- Monitor API usage and costs
- Test with production-like data

---

## See Also

- [Deploy to Production](deploy-to-production.md) — Docker and serverless deployment
- [Save and Load a Prompter](save-and-load.md) — Persistence patterns
- [Configure Optimizations](configure-optimizations.md) — Optimization tuning
