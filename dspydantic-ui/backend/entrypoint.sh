#!/bin/bash
set -e

# Install dspydantic from parent directory if available
if [ -d "/workspace/src/dspydantic" ] && [ -f "/workspace/pyproject.toml" ]; then
    echo "Installing dspydantic from workspace..."
    pip install -e /workspace
elif [ -d "../src/dspydantic" ] && [ -f "../pyproject.toml" ]; then
    echo "Installing dspydantic from parent directory..."
    pip install -e ../
fi

# Run the main command
exec "$@"
