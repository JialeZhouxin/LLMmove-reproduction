#!/bin/bash
set -e
cd "$(dirname "$0")"

# Install dependencies
uv pip install -r requirements.txt

# Run evaluation
python main.py "$@"
