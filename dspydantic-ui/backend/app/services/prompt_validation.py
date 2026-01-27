"""Utilities for validating prompts."""
from jinja2 import Environment, Template, TemplateSyntaxError
from typing import Tuple


def validate_jinja2_template(template: str | None) -> Tuple[bool, str | None]:
    """Validate that a string is a valid Jinja2 template.
    
    Args:
        template: The template string to validate.
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    if template is None or not template.strip():
        return True, None
    
    try:
        env = Environment()
        env.parse(template)
        return True, None
    except TemplateSyntaxError as e:
        return False, f"Invalid Jinja2 template: {str(e)}"
    except Exception as e:
        return False, f"Template validation error: {str(e)}"


def validate_system_prompt(prompt: str | None) -> Tuple[bool, str | None]:
    """Validate a system prompt.
    
    Args:
        prompt: The system prompt to validate.
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    if prompt is None:
        return True, None
    
    # Basic validation: check if it's not empty
    if not prompt.strip():
        return False, "System prompt cannot be empty"
    
    # Could add more validation here (length, content, etc.)
    return True, None


def render_jinja2_template(template: str, context: dict) -> str:
    """Render a Jinja2 template with the given context.
    
    Args:
        template: The Jinja2 template string.
        context: Dictionary of variables to use in the template.
        
    Returns:
        Rendered template string.
    """
    env = Environment()
    jinja_template = env.from_string(template)
    return jinja_template.render(**context)
