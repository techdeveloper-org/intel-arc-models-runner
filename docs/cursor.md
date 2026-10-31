# Using Cursor with arc-runner

> **⚠️ Cursor Subscription Required for Custom / Local Models.**
> Cursor requires a **paid Pro subscription ($20/month)** to use any custom model via a custom API key — including local models served by arc-runner. On the **free plan**, all AI features (Chat, Edit, Agent, Tab autocomplete) show the error *"Agent and Edit rely on custom models that require a paid subscription"* and are **blocked**. The arc-runner server itself works correctly; this is a Cursor-side plan restriction.
>
> **Free alternative:** [Continue.dev](https://www.continue.dev) is a free VS Code / JetBrains extension that supports Chat, Edit, and Agent with arc-runner's OpenAI-compatible endpoint — no subscription needed. See [section 2b](#2b-continue-dev-free-alternative-no-subscription) below for the setup.

`arc-runner serve` exposes `POST /v1/chat/completions` (OpenAI-compatible endpoint). If you have a **Cursor Pro** subscription, Cursor can be pointed at it just like you would point it at a local Ollama or LM Studio server. The model is a local OpenVINO model running on your Intel Arc GPU or NPU — no OpenAI API key or internet connection is needed for inference.

**Status of this guide.** The server-side OpenAI-compatible API is covered by automated tests that use fake engines. Cursor itself was **not run** while this was written, and no real Cursor session was started against this server. Everything that depends on Cursor's runtime behaviour is marked **UNVERIFIED**. Cursor's AI features and settings UI may change; verify against the [Cursor documentation](https://docs.cursor.com) for your installed version.

---

## 0. Prerequisites

1. **Install arc-runner.** No release is published yet on PyPI, so `pip install arc-runner` does not work. Follow the README section "Option 3: Windows Native Setup" (`.\\install.bat` from a checkout), then open a **new** terminal. If `arc-runner` is not found, use `python -m arc_runner.cli.main --version`.
2. **Install Cursor.** Download from [cursor.com](https://www.cursor.com). This repository does not install it.
3. **Pull a model.** Models must be pulled before the server can serve them (internet required on first pull):

```bash
arc-runner pull llama-3.2-3b
```

---

## 1. Start the Server

Pull a model once, then start the server. Both commands must share the same working directory (or use `--dir` / `--model-dir` flags):

```bash
arc-runner pull llama-3.2-3b
arc-runner models
arc-runner serve --host 127.0.0.1 --port 11434 --default-model llama-3.2-3b
```

**Port conflict with Ollama:** Ollama commonly listens on port `11434`. Only one process can bind the same host and port. If Ollama is running:

- Either **stop Ollama** and use port `11434` for arc-runner, or
- Start arc-runner on **a different port** (e.g., `11435`) and use that port in all Cursor settings:

```bash
arc-runner serve --host 127.0.0.1 --port 11435 --default-model llama-3.2-3b
```

**Health check** (no model required, no key required):

```bash
curl http://127.0.0.1:11434/health
# → {"status":"ok"}
```

**List available models** (use the `id` value in Cursor's model name field):

```bash
curl http://127.0.0.1:11434/v1/models
```

Leave the server running in a separate terminal for the duration of your Cursor session.

---

## 2. Configure Cursor to Use arc-runner

> **Cursor Pro required.** The steps below require a paid Cursor subscription. Free plan users should skip to [section 2b](#2b-continue-dev-free-alternative-no-subscription).

Cursor uses the OpenAI-compatible API. Configuration is done inside Cursor's **Settings → Models** (or **Preferences → Cursor Settings** depending on your version).

### Step-by-step (UNVERIFIED — Cursor UI may differ)

1. Open Cursor. Go to **Settings** (`Ctrl+Shift+J`) → **Models** tab.
2. Enter the following:

| Field | Value |
|:---|:---|
| **API Base URL / Base URL** | `http://127.0.0.1:11434/v1` |
| **API Key** | `local` (any non-empty string works unless `ARC_API_KEY` is set on the server) |
| **Model Name** | `qwen-2.5-1.5b` (or whichever model you pulled; check `arc-runner models`) |

3. **Save / Apply** the settings.
4. In the Cursor chat panel (`Ctrl+L`), select the custom model from the model picker.

> **Note:** Cursor appends `/chat/completions` to the base URL you provide. Make sure the base URL ends with `/v1` (not `/v1/`).

---

## 2b. Continue.dev — Free Alternative (No Subscription)

[Continue.dev](https://www.continue.dev) is a free, open-source VS Code extension that connects to any OpenAI-compatible local server. It supports **Chat, Edit (inline code), and Agent** modes with no subscription required.

### Install Continue in VS Code

1. Open VS Code → `Ctrl+Shift+X` (Extensions) → search **`Continue`** → **Install**.
2. A Continue icon appears in the left sidebar. Click it.
3. Open `~/.continue/config.json` (Continue creates this on first launch) and add:

```json
{
  "models": [
    {
      "title": "Qwen 2.5 1.5B (Arc GPU)",
      "provider": "openai",
      "model": "qwen-2.5-1.5b",
      "apiBase": "http://127.0.0.1:11434/v1",
      "apiKey": "local"
    },
    {
      "title": "DeepSeek R1 1.5B (Arc GPU)",
      "provider": "openai",
      "model": "deepseek-r1-1.5b",
      "apiBase": "http://127.0.0.1:11434/v1",
      "apiKey": "local"
    }
  ]
}
```

4. Save the file. Continue will reload automatically.
5. Make sure arc-runner is running (`arc-runner serve ...`), then use Continue's Chat, Edit, and Agent features freely.

### Continue Feature Availability (all free)

| Feature | How to use |
|:---|:---|
| **Chat** | Continue sidebar → type your question |
| **Edit (inline)** | Select code → `Ctrl+I` → describe the change |
| **Agent** | Continue sidebar → Agent tab → describe the task |
| **Codebase indexing** | Continue indexes your project for context |

---

## 3. Smoke Test with curl

Before opening Cursor, verify the server answers `POST /v1/chat/completions`:

**Bash (blocking):**

```bash
curl -s http://127.0.0.1:11434/v1/chat/completions \
  -H "content-type: application/json" \
  -H "authorization: Bearer local" \
  -d '{
    "model": "llama-3.2-3b",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Say hi"}]
  }'
```

**Expected shape** (verified with a fake engine only; real-engine text and token counts are UNVERIFIED):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "llama-3.2-3b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hi! ..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {}
}
```

**Bash (streaming — you should see `data: {"choices":[...]}` lines followed by `data: [DONE]`):**

```bash
curl -sN http://127.0.0.1:11434/v1/chat/completions \
  -H "content-type: application/json" \
  -H "authorization: Bearer local" \
  -d '{
    "model": "llama-3.2-3b",
    "max_tokens": 64,
    "stream": true,
    "messages": [{"role": "user", "content": "Say hi"}]
  }'
```

**Windows PowerShell** (UNVERIFIED — not run on Windows; write the body to a file to avoid quoting issues; use `curl.exe`, not the `curl` alias):

```powershell
Set-Content -Path body.json -Encoding ascii -Value '{"model":"llama-3.2-3b","max_tokens":64,"messages":[{"role":"user","content":"Say hi"}]}'
curl.exe -s http://127.0.0.1:11434/v1/chat/completions `
  -H "content-type: application/json" `
  -H "authorization: Bearer local" `
  --data-binary "@body.json"
```

If the smoke test passes, Cursor should work with the same endpoint.

---

## 4. Context Window

The server enforces a context-window budget. For local models on a laptop with shared GPU/system memory, a large context window can exhaust memory. Control it with the `ARC_CONTEXT_WINDOW` environment variable (set **before** starting the server):

```bash
# Bash
ARC_CONTEXT_WINDOW=8192 arc-runner serve --default-model llama-3.2-3b

# Windows PowerShell
$env:ARC_CONTEXT_WINDOW = "8192"
arc-runner serve --default-model llama-3.2-3b --port 11434
```

The full context-window resolution order and memory notes are in [docs/claude-code.md — The context window](claude-code.md#the-context-window) (the same engine logic applies here).

---

## 5. What Is Implemented (OpenAI-compatible routes)

| Route | Status |
|:---|:---|
| `POST /v1/chat/completions` | Blocking and streaming (`stream: true`). Message roles `system`, `user`, `assistant` are supported. |
| `GET /v1/models` | Returns the list of locally pulled models. |
| `POST /v1/embeddings` | NPU embedding path (requires an embeddings model; not all pulled models support it). |
| `GET /health` | `{"status":"ok"}` — no key required. |
| `GET /` | `"Ollama is running"` — compatibility probe. |

Cursor primarily uses `POST /v1/chat/completions`. The server also answers `GET /v1/models` if Cursor queries it for model discovery (UNVERIFIED: whether Cursor calls this automatically).

---

## 6. Not Supported / Limitations

### Cursor Plan Restriction — Agent and Edit Require Paid Plan

> **Important:** Cursor's **Agent mode** and **Edit mode** (Composer) require a **paid Cursor subscription ($20/month)** even when using a custom local model via a custom API key. On the Cursor Free plan, only **Chat mode** (`Ctrl+L`) works with a custom OpenAI-compatible endpoint. If you see the error *"Agent and Edit rely on custom models that require a paid subscription"*, this is expected Cursor behaviour and is unrelated to arc-runner.

**Workaround — use Continue.dev instead (fully free, Agent + Edit supported):**

[Continue.dev](https://www.continue.dev) is a free VS Code / JetBrains extension that connects to any OpenAI-compatible local server with no subscription required. Add this to your `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Qwen 2.5 1.5B (Arc GPU)",
      "provider": "openai",
      "model": "qwen-2.5-1.5b",
      "apiBase": "http://127.0.0.1:11434/v1",
      "apiKey": "local"
    }
  ]
}
```

Then start arc-runner as usual and use Continue's Chat, Edit, and Agent features freely.

### Other Limitations

- **Images / vision**: `image_url` content blocks are not supported by this route. A conversation with an image will keep failing. Clear the chat context in Cursor to recover.
- **Function / tool calling quality**: Tool calling depends entirely on the model. Small local models (1.5B–3B) often ignore the expected format. Cursor's AI features (Composer, Tab, Agent) that rely on tool calls may produce degraded results compared to a full-size model. No measurement has been made for Cursor-specific tool-use quality with any model in this repository (UNVERIFIED).
- **Prompt caching**: `cache_control` is accepted but ignored; no prefix caching occurs.
- **`max_tokens` above 32768**: Returns `400 invalid_request_error`.
- **Cursor Composer / Agent**: These features send large system prompts with many tool definitions. The time to first token with Cursor's full prompt on a 15 GB laptop with a shared GPU has **not been measured**; no number is given here.

---

## 7. Security Notes

By default, arc-runner binds to `127.0.0.1` (loopback only). Cursor running on the same machine connects to `http://127.0.0.1:11434/v1` without any network exposure.

If you set `ARC_API_KEY` on the server, every API route requires that key in `Authorization: Bearer <key>` or `x-api-key: <key>`. In Cursor's API Key field, enter the same value.

Do **not** bind arc-runner to `0.0.0.0` on a shared or public network unless you understand the access implications. See [docs/deployment.md](deployment.md) for `ARC_ALLOWED_HOSTS` settings.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|:---|:---|
| Cursor says "Invalid API Key" or "Unauthorized" | If `ARC_API_KEY` is set on the server, enter the same value in Cursor's API Key field. Otherwise, enter `local` (any non-empty string). |
| `404 not_found_error: model: X` | The model name in Cursor does not match any locally pulled model. Run `arc-runner models` or `curl .../v1/models` to see valid names. |
| `400 prompt is too long` | The combined prompt + max_tokens exceeds the context window. Lower Cursor's max output setting or increase `ARC_CONTEXT_WINDOW` (memory permitting). |
| `400 max_tokens: ... exceeds the maximum of 32768` | Lower the max tokens setting in Cursor. |
| Cursor chat hangs / no response | The model is being compiled on first use — this can take tens of seconds. Check server logs. Also confirm the base URL ends with `/v1` (no trailing slash in some Cursor versions). |
| Port 11434 in use | Ollama is running. Use `--port 11435` for arc-runner and update Cursor's Base URL to `http://127.0.0.1:11435/v1`. |
| Connection refused | arc-runner server is not running. Start it and confirm `curl http://127.0.0.1:11434/health` returns `{"status":"ok"}`. |
| Tab / Autocomplete does not trigger | Cursor Tab uses a different model slot. Ensure the custom model is selected in the Tab settings tab (if configurable) and the server is running. UNVERIFIED: Cursor Tab behaviour with custom OpenAI-compatible endpoints. |

---

## 9. UNVERIFIED Summary

The following items were **not verified** while this guide was written:

- Cursor's runtime behaviour against this server: Cursor was not opened and no real session was run.
- Whether Cursor accepts an empty `usage` object in `POST /v1/chat/completions` responses.
- Cursor's exact UI field names and layout (these change between Cursor releases).
- Whether Cursor calls `GET /v1/models` automatically for model discovery.
- Tool-calling quality (Composer, Agent, Tab) with any local model against this server.
- Time to first token with Cursor's full prompt on Intel Core Ultra hardware.
- Whether `ARC_CONTEXT_WINDOW` must be set for Cursor to work reliably (depends on model config.json and available GPU memory).
- Real-model streaming: tests use fake engines; the OpenVINO engine stream was not exercised through this route with Cursor.
