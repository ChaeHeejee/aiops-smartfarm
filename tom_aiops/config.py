"""Application configuration for tom_aiops."""

import os

# Set via environment variable: export OPENAI_API_KEY=sk-...
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
