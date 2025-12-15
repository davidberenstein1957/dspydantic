"""Evaluation functions for different optimization backends."""

from dspydantic.evaluators.dspy import default_evaluate_fn, default_judge_fn

__all__ = ["default_evaluate_fn", "default_judge_fn"]

# Optional GLiNER2 imports - only available if gliner2 is installed
try:
    from dspydantic.evaluators.gliner import default_gliner_evaluate_fn

    __all__.append("default_gliner_evaluate_fn")
except ImportError:
    # GLiNER2 not available
    pass
