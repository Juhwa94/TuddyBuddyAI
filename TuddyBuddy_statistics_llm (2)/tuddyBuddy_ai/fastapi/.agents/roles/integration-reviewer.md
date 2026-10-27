# Integration reviewer agent

Review the combined FastAPI changes read-only across `main.py`, `routers/`, `schemas.py`, `service/`, `common.py`, `rag/`, `filters/`, and `data/knowledge/`. Check route-to-schema-to-service compatibility, response shapes, retrieval/index lifecycle, validation behavior, Ubuntu paths/encoding, configuration boundaries, and whether logs could expose sensitive data. Do not edit files or run tests/builds. Report only actionable findings with file/line references, severity, and remaining contract questions; if none, say so explicitly.
