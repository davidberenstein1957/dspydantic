"""Extractors for different optimization backends."""

from dspydantic.extractors.dspy import (
    apply_optimized_descriptions,
    create_optimized_model,
    extract_field_descriptions,
    extract_field_types,
)

__all__ = [
    "extract_field_descriptions",
    "extract_field_types",
    "apply_optimized_descriptions",
    "create_optimized_model",
]

# Optional GLiNER2 imports - only available if gliner2 is installed
try:
    from dspydantic.extractors.gliner import (
        apply_optimized_gliner_descriptions,
        extract_gliner_descriptions,
    )

    __all__.extend(
        [
            "extract_gliner_descriptions",
            "apply_optimized_gliner_descriptions",
        ]
    )
except ImportError:
    # GLiNER2 not available
    pass
