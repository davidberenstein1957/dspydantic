"""DSPy modules for optimization."""

from dspydantic.modules.dspy import PydanticOptimizerModule

__all__ = ["PydanticOptimizerModule"]

# Optional GLiNER2 imports - only available if gliner2 is installed
try:
    from dspydantic.modules.gliner import GLiNER2OptimizerModule

    __all__.append("GLiNER2OptimizerModule")
except ImportError:
    # GLiNER2 not available
    pass
