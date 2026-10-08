# Local model

The application uses a **small local instruction model (~1B parameters) on CPU**. No cloud LLM is ever called.
Model files are not included in the repository (they are several hundred MB); place one here yourself.

## Option A — GGUF file with llama.cpp (default)

1. Download a Q4 quantisation of **Llama 3.2 1B Instruct** in GGUF format, e.g. `Llama-3.2-1B-Instruct-Q4_K_M.gguf`
   (≈ 0.8 GB) from a GGUF publisher on Hugging Face (for example the `bartowski` or `unsloth` repositories).
   Check the Llama 3.2 licence terms before use.
2. Save it as:

   ```
   models/model.gguf
   ```

   or keep the original name and set `LOCAL_LLM_MODEL_PATH` in `backend/.env`.
3. Install the inference library into the backend virtualenv:

   ```
   pip install -r requirements-llm.txt
   # Windows/macOS without a C++ compiler — use prebuilt CPU wheels:
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   ```

4. Restart the backend. `GET /api/llm/status` (and the banner in the UI) should report the model as ready.

RAM: a 1B Q4 model needs roughly 1–1.5 GB, so 8 GB machines work; 16 GB is comfortable.

## Option B — Ollama

```
ollama pull llama3.2:1b
```

Then in `backend/.env`:

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:1b
```

## No model yet?

The app still starts and runs analyses using deterministic keyword/regex extraction. It shows:

> Local LLM model is not configured. Place the GGUF model in models/ or configure an Ollama model.

Every recommendation produced this way has confidence capped at Medium and is marked *Requires review*.
