# FastAPI LLM team guide

This directory contains the FastAPI LLM service for TuddyBuddy. Keep changes scoped to this service and follow existing Python, FastAPI, Pydantic, and LangChain patterns. The parent `tuddyBuddy_ai` directory is the application root; its `.env` is loaded by `common.py`. Never read, print, copy, or commit secret values from `.env`.

## Team roles

Use the focused role briefs in `.agents/roles/` when delegating. Assign independent work to separate roles, agree on request/response contracts before cross-layer changes, and have the integration reviewer inspect the combined changes read-only. The coordinator owns shared files and integration decisions.

- `api.md`: FastAPI routes, request handling, HTTP behavior, and Java-facing API contracts.
- `llm-service.md`: prompts, provider/model integration, structured output, and LLM-side safeguards.
- `rag.md`: JSONL knowledge corpus, Ubuntu index generation, and retrieval.
- `response-filter.md`: LLM response validation and output constraints.
- `schemas-data.md`: Pydantic schemas, payload compatibility, validation, and data-shape review.
- `integration-reviewer.md`: read-only cross-layer consistency and regression review.

## Project map

- `main.py`: application setup, `/statistics` and `/prediction` routes, exception handling.
- `routers/`: route modules (currently `analysis.py` is a placeholder module).
- `schemas.py`: request and structured response models.
- `service/llm_service.py`: LLM calls, prompts, and response construction.
- `service/category_grouping.py`: deterministic category normalization, grouping, and duplicate-stat aggregation.
- `common.py`: environment loading, model configuration, and provider factory.
- `rag/`: knowledge ingestion and retrieval; source data lives under `data/knowledge/`.
- `filters/`: response validation before returning from service functions.

Preserve user changes. Trace callers and current payloads before changing contracts. Do not run tests or builds unless the user asks for verification.

