$ErrorActionPreference = "Stop"

pytest tests/test_supplement_query_understanding.py `
       tests/test_supplement_response_composer.py `
       tests/test_supplement_kb_schema.py `
       tests/test_supplement_engine_degraded.py `
       -m "not slow and not llm and not smoke"
