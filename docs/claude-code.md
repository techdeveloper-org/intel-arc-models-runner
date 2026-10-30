# Using Claude Code with arc-runner

`arc-runner serve` answers the Anthropic Messages API (`POST /v1/messages`), so Claude Code can be pointed at it the way people point it at Ollama. The model is a local OpenVINO model, not a Claude model.

**Status of this guide.** The server side is covered by automated tests that use fake engines and by a run of the real app under uvicorn with raw sockets. Claude Code itself was **not run** while this was written, and no real model was involved. Everything that depends on Claude Code's behaviour or on a real model is marked UNVERIFIED. Anthropic does not support routing Claude Code to non-Claude models; this works only as far as the wire protocol matches.

## Read this first: the context window

The engines in this repository enforce a context window of 4096 tokens (`DEFAULT_CONTEXT_WINDOW` in `src/arc_runner/core/constants.py`); no setting or environment variable changes it. A request fails when `prompt tokens + max_tokens` exceeds it. Claude Code sends its own system prompt plus the schemas of all its tools with every request, and by default asks for `max_tokens` 32000 for an unknown model. Two consequences, both read from the code and neither tried against Claude Code:

- Every request is expected to answer `400 invalid_request_error` with a message beginning `prompt is too long`, until the window is made configurable in the engine. That change is outside this API layer.
- Lowering `CLAUDE_CODE_MAX_OUTPUT_TOKENS` helps only the `max_tokens` half of the check.

The exact message text Claude Code keys its automatic compaction on is UNVERIFIED; the runner uses Anthropic's style, `prompt is too long: N tokens (prompt plus max_tokens) > M maximum context window`. If Claude Code does not compact by itself, run `/compact` or `/clear`.

## 1. Start the server

Pull a model once, then serve it (see the README for model names):

```bash
arc-runner pull llama-3.2-3b
arc-runner models
arc-runner serve --host 127.0.0.1 --port 11434 --default-model llama-3.2-3b
```

If Ollama already listens on 11434, use another port such as 11435 and change the URL below to match. Find the exact model name Claude Code must send:

```bash
curl http://127.0.0.1:11434/v1/models
```

Use a value of `data[].id`; an unknown name returns `404 not_found_error` listing the local models. There is no silent fallback to another model.

## 2. Point Claude Code at it

Bash:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:11434
export ANTHROPIC_AUTH_TOKEN=local
export ANTHROPIC_MODEL=llama-3.2-3b
export ANTHROPIC_DEFAULT_HAIKU_MODEL=llama-3.2-3b
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=2048
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
claude
```

Windows PowerShell:

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:11434"
$env:ANTHROPIC_AUTH_TOKEN = "local"
$env:ANTHROPIC_MODEL = "llama-3.2-3b"
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = "llama-3.2-3b"
$env:CLAUDE_CODE_MAX_OUTPUT_TOKENS = "2048"
$env:CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS = "1"
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
$env:CLAUDE_CODE_ATTRIBUTION_HEADER = "0"
claude
```

| Variable | Why |
|:---|:---|
| `ANTHROPIC_BASE_URL` | Host and port only, no `/v1`; Claude Code appends `/v1/messages` itself. |
| `ANTHROPIC_AUTH_TOKEN` | Sent as `Authorization: Bearer`. Any value works unless `ARC_API_KEY` is set (below). Preferred over `ANTHROPIC_API_KEY`, which makes interactive Claude Code ask once to approve the key. |
| `ANTHROPIC_MODEL` | The local model name from `/v1/models`. Claude Code does not validate it at start; a typo fails on the first request. |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Background tasks may use the "haiku" alias; pin it to the local model. `ANTHROPIC_SMALL_FAST_MODEL` is deprecated. |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | Claude Code assumes 32000 for unknown models. Set it below the engine's context window. The runner rejects `max_tokens` above 32768 with a 400. |
| `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | Claude Code assumes a 200K context for unknown model names and cannot learn a smaller one from the server. The documented minimum is 100000, so it cannot match a smaller local window; `/compact` is the recovery. |
| `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` | Strips beta fields. The runner ignores them anyway; this only avoids sending them. |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | No telemetry or update checks while working locally. |
| `CLAUDE_CODE_ATTRIBUTION_HEADER=0` | Omits a changing attribution block at the start of the system prompt, which would defeat any prefix reuse. |

Settings in `~/.claude/settings.json` (`env` block) override shell exports.

### With `ARC_API_KEY`

When the server runs with `ARC_API_KEY` set, every route except `/`, `/docs`, `/openapi.json`, `/health` and `/static` needs the key. For `/v1/messages` the key may arrive in `Authorization: Bearer <key>` or in `x-api-key: <key>`; a match in either header is accepted (both are always compared, in constant time). Set `ANTHROPIC_AUTH_TOKEN` (or `ANTHROPIC_API_KEY`) to the same value. A wrong or missing key returns `401 authentication_error`. The `HEAD /api/hello` startup probe also needs the key; Claude Code tolerates a rejected probe (UNVERIFIED for 401 specifically).

## 3. Smoke tests with curl

Bash, blocking:

```bash
curl -s "http://127.0.0.1:11434/v1/messages?beta=true" \
  -H "content-type: application/json" -H "anthropic-version: 2023-06-01" -H "authorization: Bearer local" \
  -d '{"model":"llama-3.2-3b","max_tokens":64,"messages":[{"role":"user","content":"Say hi"}]}'
```

Bash, streaming (`-N` turns off curl's buffering; you should see `event: message_start`, then deltas, then `event: message_stop`):

```bash
curl -sN "http://127.0.0.1:11434/v1/messages" \
  -H "content-type: application/json" -H "authorization: Bearer local" \
  -d '{"model":"llama-3.2-3b","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"Say hi"}]}'
```

Token counting (measured with the engine tokenizer on the exact prompt, never estimated):

```bash
curl -s http://127.0.0.1:11434/v1/messages/count_tokens \
  -H "content-type: application/json" \
  -d '{"model":"llama-3.2-3b","messages":[{"role":"user","content":"Say hi"}]}'
```

PowerShell (write the body to a file to avoid quoting problems; use `curl.exe`, not the `curl` alias):

```powershell
Set-Content -Path body.json -Encoding ascii -Value '{"model":"llama-3.2-3b","max_tokens":64,"messages":[{"role":"user","content":"Say hi"}]}'
curl.exe -s http://127.0.0.1:11434/v1/messages -H "content-type: application/json" -H "authorization: Bearer local" --data-binary "@body.json"

Set-Content -Path stream.json -Encoding ascii -Value '{"model":"llama-3.2-3b","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"Say hi"}]}'
curl.exe -sN http://127.0.0.1:11434/v1/messages -H "content-type: application/json" -H "authorization: Bearer local" --data-binary "@stream.json"
```

## 4. What is implemented

- `POST /v1/messages`, blocking and streaming; the query string (`?beta=true`) is ignored. Message ids are `msg_...`.
- Request fields: `model`, `max_tokens` (1 to 32768, required), `messages` (string or block content: `text`, `tool_use`, `tool_result`; `role: "system"` entries inside `messages` are accepted), `system` (string or text blocks), `stream`, `temperature` (0 to 2), `top_p`, `top_k` (0 means unset), `stop_sequences`, `tools`, `tool_choice`.
- Unknown fields are never rejected: `thinking` (including `adaptive`), `output_config`, `context_management`, `metadata`, `cache_control` anywhere, beta tool fields. `anthropic-beta` and other `anthropic-*` headers are ignored. Replayed `thinking` blocks in history are dropped.
- Streaming events in this order: `message_start`, per block `content_block_start`, `content_block_delta`, `content_block_stop`, then `message_delta` (stop reason and usage), `message_stop`. `ping` events are sent every 10 seconds while the engine is silent (slow prefill) so Claude Code's idle watchdog does not abort. No `thinking` blocks are ever produced. A failure after the stream started is sent as `event: error` and the stream ends without `message_stop`.
- Errors use `{"type":"error","error":{"type","message"},"request_id"}` with 400 `invalid_request_error`, 401 `authentication_error`, 404 `not_found_error`, 413 `request_too_large` (body above 64 MiB), 429, 500 `api_error`, 504 `timeout_error`, 529 `overloaded_error`. A prompt that does not fit the context window returns HTTP 400 before any stream bytes are sent.
- `HEAD` and `GET /api/hello` return 200 (Claude Code's connection probe).
- `POST /v1/messages/count_tokens` returns `{"input_tokens": N}` measured by the engine tokenizer. If the loaded engine has no tokenizer it returns `404 not_found_error` ("token counting is unavailable") and no number; Claude Code falls back to its own estimate (documented behaviour for a missing endpoint).
- Usage: `usage.input_tokens` and `usage.output_tokens` appear only when the engine measured them. When it did not, the `usage` object is empty rather than zero-filled. Whether Claude Code accepts an empty `usage` is UNVERIFIED.
- Engine access per model is serialised: a second request waits for the first. When the client disconnects the engine stream is closed at the next token; work already inside a prefill cannot be interrupted.

## 5. Tools

Claude Code needs tool calls to read and edit files. The runner supports them as follows, and tool quality depends entirely on the model:

1. The Anthropic `tools` are written into the system prompt as a compact, deterministic instruction (one JSON line per tool: name, description, `input_schema`). If an engine ever provides `render_chat_with_tools(messages, tools)` and its tokenizer template supports tools, that is used instead; no engine in this repository provides it today, so the instruction is always used.
2. The model is told to answer a call as `<tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>`. Only that exact format is accepted: the name must be a declared tool, `arguments` must be a JSON object, no other keys. Prose such as "I would call X", partial or malformed JSON, an unclosed tag, an unknown tool, or one bad block among several turns the whole answer into plain text. Nothing is ever invented from prose. Valid calls become `tool_use` blocks (`toolu_...` ids, `stop_reason: tool_use`; streamed as `content_block_start` with empty `input`, then `input_json_delta`).
3. `tool_use` and `tool_result` blocks in the history are rendered back as `<tool_call>` and `<tool_response id=... name=...>` text.
4. `tool_choice`: `{"type":"none"}` hides the tools and disables parsing. `any` and `tool` are accepted but not enforced.

Small models often ignore the format or produce bad arguments (UNVERIFIED for the models in this repository; not measured). Claude Code's tool list is large, so the system prompt is large; the time to first token with such a prompt on a 15 GB laptop has not been measured and no number is given here. Measured results exist only in the committed files under `docs/benchmarks/`.

## 6. Not supported

- Images and documents (`image`, `document` blocks, also inside `tool_result`): `400 invalid_request_error`, because there is no vision path behind this route. A conversation that already contains an image keeps failing until `/clear` or `/compact` removes it.
- Unknown block types in messages: 400 naming the type.
- `thinking` output, prompt caching (`cache_control` is ignored, so nothing is cached), citations, batches, the models-beta fields.
- `stop_reason: stop_sequence`: the engine does not report which stop string ended the output, so it is reported as `end_turn`. `max_tokens` is reported when the engine says it stopped for length (streaming) or when the measured completion count reached `max_tokens` (blocking).
- Assistant prefill (a trailing assistant message is rendered as a finished turn, not continued).
- `max_tokens` above 32768: `400` that names the limit.

## 7. Troubleshooting

| Symptom | Cause and fix |
|:---|:---|
| `404 not_found_error: model: X (not found; available local models: ...)` | `ANTHROPIC_MODEL` is not a name from `/v1/models`. Fix the variable; also set `ANTHROPIC_DEFAULT_HAIKU_MODEL`. |
| `401 authentication_error` | `ARC_API_KEY` is set on the server. Set `ANTHROPIC_AUTH_TOKEN` (or `ANTHROPIC_API_KEY`) to the same value in the shell that starts `claude`; `~/.claude/settings.json` `env` overrides the shell. |
| `400 prompt is too long: ...` | The prompt plus `max_tokens` does not fit the engine's context window (see the top of this page). Lower `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, run `/compact` or `/clear`. A 4096-token window cannot hold Claude Code's own prompt. |
| `400 max_tokens: ... exceeds the maximum of 32768` | Lower `CLAUDE_CODE_MAX_OUTPUT_TOKENS`. |
| `400 ... content is not supported by this runner` | An image or document is in the request. `/clear` or `/compact`. |
| Tool calls are ignored, or code is printed instead of edited | The model did not answer in the `<tool_call>` format, so the answer was returned as text. Try a larger or tool-tuned model; check `curl` with a `tools` array to see the raw text. |
| "API returned an empty or malformed response" | The URL probably points at something else (a proxy, another service on the port). `curl` the smoke test above against the same URL. |
| Long silence on the first request | The model is compiled on first use. Headers are not sent until the engine produces its first event or ten seconds pass; Claude Code waits up to `API_TIMEOUT_MS` (default 600000) for headers. |
| Port 11434 in use | Ollama is running. Use `--port 11435` and update `ANTHROPIC_BASE_URL`. |

## 8. UNVERIFIED summary

- Everything about Claude Code's runtime behaviour against this server: it was not run. The protocol details came from Anthropic's public documentation as read on 2026-10-30.
- That Claude Code accepts an empty `usage` object, a 404 from `count_tokens` and a 401 from `/api/hello`.
- The exact wording Claude Code recognises for automatic compaction after a context-length 400.
- The quality of tool calling for any model here, and time to first token with Claude Code's system prompt on any hardware.
- Real-model streaming behaviour: tests use fake engines; the OpenVINO engine's stream was not exercised through this route.
