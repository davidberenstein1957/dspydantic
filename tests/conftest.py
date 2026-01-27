"""Shared pytest fixtures."""

import os

import dspy
import pytest


@pytest.fixture
def lm():
    """Configure DSPy with gpt-4.1-mini.

    Uses OPENAI_API_KEY when set (for integration tests); otherwise "test-key"
    (for unit tests that only need dspy configured).
    """
    api_key = os.getenv("OPENAI_API_KEY", "test-key")
    lm_ = dspy.LM("openai/gpt-4.1-mini", api_key=api_key)
    dspy.configure(lm=lm_)
    return lm_
