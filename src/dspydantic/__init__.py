"""dspydantic - Optimize Pydantic model field descriptions using DSPy."""

from dspydantic.extractor import (
    apply_optimized_descriptions,
    create_optimized_model,
    extract_field_descriptions,
)
from dspydantic.optimizer import PydanticOptimizer
from dspydantic.types import Example, OptimizationResult
from dspydantic.utils import (
    image_to_base64,
    pdf_to_base64_images,
    prepare_input_data,
)

__version__ = "0.0.6"
__all__ = [
    "PydanticOptimizer",
    "Example",
    "OptimizationResult",
    "extract_field_descriptions",
    "apply_optimized_descriptions",
    "create_optimized_model",
    "prepare_input_data",
    "image_to_base64",
    "pdf_to_base64_images",
]

