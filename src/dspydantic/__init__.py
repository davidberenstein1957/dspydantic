"""dspydantic - Optimize Pydantic model field descriptions using DSPy."""

from dspydantic.extractors import (
    apply_optimized_descriptions,
    create_optimized_model,
    extract_field_descriptions,
)
from dspydantic.optimizers import PydanticOptimizer
from dspydantic.types import Example, OptimizationResult
from dspydantic.utils import (
    image_to_base64,
    pdf_to_base64_images,
    prepare_input_data,
)

__version__ = "0.0.7"
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

# Optional GLiNER2 imports - only available if gliner2 is installed
try:
    from dspydantic.extractors import (
        apply_optimized_gliner_descriptions,
        extract_gliner_descriptions,
    )
    from dspydantic.optimizers import GLiNER2SchemaOptimizer

    __all__.extend(
        [
            "GLiNER2SchemaOptimizer",
            "extract_gliner_descriptions",
            "apply_optimized_gliner_descriptions",
        ]
    )
except ImportError:
    # GLiNER2 not available - users can install with: pip install dspydantic[gliner]
    pass
