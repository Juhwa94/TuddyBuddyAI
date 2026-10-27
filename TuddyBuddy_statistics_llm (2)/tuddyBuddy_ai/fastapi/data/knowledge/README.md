# FastAPI RAG knowledge data

The source of truth is UTF-8 JSONL under `data/knowledge/`, one standalone knowledge record per line. Keep prediction rules in `prediction_policy.jsonl` and study guidance in `study_guidance.jsonl`. These are project-owned guidance records, not user score history or model-training examples.

Category grouping happens in Python before the LLM call. Records merge when they share a `categoryId`, a punctuation/spacing/case-normalized `categoryName`, or an explicit `categoryGroup`. Supply the same `categoryGroup` when differently named categories are known to be semantically equivalent; Python does not infer semantic similarity from the RAG corpus. `/statistics` needs category metadata in each quiz log for category-specific analysis; old payloads without it are analyzed as one `전체` group.

From the `fastapi` directory, rebuild the Chroma index from the JSONL source on the Ubuntu host with:

```bash
python -m rag.ingest
```

Each run replaces the two managed collections from their JSONL sources, including removal of deleted records. Set `RAG_PERSIST_DIR` to a writable persistent directory on Ubuntu when the application directory is replaced during deployment. If the index is absent or retrieval fails, the API continues with its existing LLM path and tells the model that retrieved guidance is unavailable. Do not copy a Windows-generated index as the deployment artifact; transfer the JSONL source and build the index in the target environment.
