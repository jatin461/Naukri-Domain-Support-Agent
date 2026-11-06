# Dependency and framework compatibility

Python **3.11.15** (`.python-version`). The system default `python3` may be newer (3.14 at development time) and is not supported by CrewAI (`Requires-Python <3.14`), so the virtual environment must be created with `python3.11`.

## Pins

Direct dependencies are pinned exactly in `requirements.txt`; the full resolved set (165 packages, `pip check` clean) is frozen in `requirements.lock`. `tests/test_environment.py` checks that the installed versions match the pins.

| Package | Pin | Role |
|---|---|---|
| `crewai` | 1.15.33 | crew, `BaseLLM` extension point |
| `autogen-agentchat`, `autogen-core` | 0.7.5 | review team (**never leave unpinned**: unpinned `autogen-agentchat` resolves to the legacy 0.2.40 line, which has no `RoundRobinGroupChat`, `StructuredMessage` or `output_content_type`) |
| `langchain-core` | 1.7.3 | `InMemoryChatMessageHistory`, `RunnableWithMessageHistory` (only `langchain-core` is needed) |
| `pydantic` | 2.12.5 | schemas |
| `fastapi` / `uvicorn` / `httpx` / `websockets` | 0.162.0 / 0.54.0 / 0.28.1 / 17.4 | API, server, `TestClient` transport, WebSocket support for the real server |
| `chromadb` | 1.5.9 | vector store |
| `sentence-transformers` | 6.2.0 | local embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 384-d) |
| `pytest` | 9.2.0 | tests |

Transitive pins worth knowing: torch 2.15.0, transformers 5.22.1, numpy 2.4.6, starlette 1.10.0. `autogen-ext` and the `langchain` meta-package are not installed.

## Framework facts the implementation relies on

- **CrewAI `BaseLLM`** has one abstract method, `call(messages, tools, callbacks, available_functions, from_task, from_agent, response_model)`. A plain subclass takes the ReAct text path (`supports_function_calling` is not defined), `tools` is `None`, and the declared tools are on `from_agent.tools` with their `args_schema`.
- **ReAct parsing pitfall.** The system prompt contains the template text `Observation: the result of the action`; the real observation arrives appended inside the previous assistant message. The mock parses only the last assistant message and builds tool arguments from each tool's declared JSON schema, never from the tool's name (`rag_lookup` takes `query`).
- **Structured output.** `Task(output_pydantic=Model)` makes `call()` receive `response_model`; the pipeline also validates the response in code.
- **CrewAI deep-copies agent LLMs and tools**, so the shared mock core and the tool invocation recorder define `__deepcopy__` to stay one shared object (otherwise counters and invocation evidence are silently lost).
- **AutoGen 0.7.5.** Agent names must be valid Python identifiers, hence `Policy_Compliance_Reviewer` and `Final_Editor` (the brief gives the hyphenated names only as examples). With `max_turns=2` the messages are the task `TextMessage`, the reviewer `TextMessage` and the editor `StructuredMessage[Verdict]`. Omitting `custom_message_types=[StructuredMessage[Verdict]]` fails with `RuntimeError("ValueError: Message type ... is not registered")`. A custom `ChatCompletionClient` needs `model_info` with `structured_output=True`.
- **ChromaDB 1.5.9.** The default embedder and telemetry are network paths, so the client uses `Settings(anonymized_telemetry=False)` and every `upsert`/`query` passes precomputed embeddings. Cosine similarity needs `metadata={"hnsw:space": "cosine"}` (similarity = 1 - distance). `upsert` is idempotent.
- **Chroma quirk.** After a collection has been read, a later small upsert was sometimes stored but never returned by queries (about 1 build in 5; 7/30 and 7/40 in repeated trials; query retries do not help). Collections therefore use `hnsw:batch_size=2` and `hnsw:sync_threshold=2` (0 misses in 150 trial builds), and `/add-document` verifies that every written chunk is searchable and re-upserts if not. A regression test covers it.
- **LangChain.** `RunnableWithMessageHistory` emits a `LangChainDeprecationWarning` pointing to LangGraph persistence. It is expected and deliberately not silenced.
- **FastAPI.** The pipeline is blocking (CrewAI, embeddings, AutoGen via `asyncio.run`), so handlers use `run_in_threadpool`; the WebSocket handler is `async def`.

## Offline and telemetry environment

Set by `src/config/settings.py`, which every entry point and `tests/conftest.py` imports before `crewai`, `chromadb` or `sentence_transformers`:

| Variable | Value | Why |
|---|---|---|
| `MOCK_LLM` | `true` | project switch; `false` raises `NotImplementedError` |
| `CREWAI_DISABLE_TELEMETRY` | `true` | required by the brief |
| `OTEL_SDK_DISABLED` | `true` | also stops OpenTelemetry exporters |
| `CREWAI_DISABLE_TRACKING` | `true` | extra CrewAI opt-out |
| `ANONYMIZED_TELEMETRY` | `False` | ChromaDB |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `1` | no model downloads in graded runs |
| `HF_HUB_DISABLE_TELEMETRY` | `1` | Hugging Face hub |
| `TOKENIZERS_PARALLELISM` | `false` | avoids fork warnings |
| `OPENAI_API_KEY` | removed | proves no hidden dependence on a key |

Zero network is enforced, not asserted: an autouse fixture in `tests/conftest.py` makes every non-loopback `socket.connect` raise, so the whole suite runs under it. The only networked step is `scripts/prefetch_model.py`, which downloads the embedding model once into the local Hugging Face cache.
