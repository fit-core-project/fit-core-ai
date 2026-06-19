$ErrorActionPreference = "Stop"

pytest -m "not slow and not integration and not llm and not chroma and not smoke"
