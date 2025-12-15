"""Optimizers for different optimization backends."""

from dspydantic.optimizers.dspy import PydanticOptimizer

__all__ = ["PydanticOptimizer"]

# Optional GLiNER2 imports - only available if gliner2 is installed
try:
    from dspydantic.optimizers.gliner import GLiNER2SchemaOptimizer

    __all__.append("GLiNER2SchemaOptimizer")
except ImportError:
    # GLiNER2 not available
    pass
