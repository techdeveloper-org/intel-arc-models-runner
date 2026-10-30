"""Pydantic v2 wire protocol schemas for Ollama and OpenAI API parity.

Defines request and response data models for autoregressive completion, conversational chat,
model tags catalog, and OpenAI v1 compatibility endpoints.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class OllamaChatMessage(BaseModel):
    """Message item in Ollama chat completion conversations."""

    role: str = Field(description="Participant role: system, user, or assistant")
    content: str = Field(description="Textual payload of message")
    images: Optional[List[str]] = Field(default=None, description="Optional base64 encoded images")


class OllamaGenerateRequest(BaseModel):
    """Payload contract for Ollama POST /api/generate endpoint."""

    model: str = Field(description="Target model identifier or path")
    prompt: str = Field(default="", description="Input prompt text")
    system: Optional[str] = Field(default=None, description="System message override")
    template: Optional[str] = Field(default=None, description="Prompt formatting template")
    context: Optional[List[int]] = Field(default=None, description="Token IDs context history")
    stream: bool = Field(default=True, description="Enable streaming NDJSON responses")
    raw: bool = Field(default=False, description="Bypass template formatting")
    images: Optional[List[str]] = Field(default=None, description="Optional base64 encoded images")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Runtime generation parameters")
    adapters: Optional[List[Dict[str, Any]]] = Field(default=None, description="Active LoRA adapters to compose")
    adapter: Optional[str] = Field(default=None, description="Single LoRA adapter name or path shortcut")


class OllamaGenerateResponse(BaseModel):
    """Response contract for Ollama POST /api/generate endpoint."""

    model: str = Field(description="Model identifier evaluated")
    created_at: str = Field(description="ISO-8601 UTC timestamp")
    response: str = Field(default="", description="Incremental or complete token string")
    done: bool = Field(description="Completion status flag")
    total_duration: Optional[int] = Field(default=None, description="Total execution duration in nanoseconds")
    load_duration: Optional[int] = Field(default=None, description="Model compilation duration in nanoseconds")
    prompt_eval_count: Optional[int] = Field(default=None, description="Number of tokens in prompt")
    prompt_eval_duration: Optional[int] = Field(default=None, description="Prefill duration in nanoseconds")
    eval_count: Optional[int] = Field(default=None, description="Number of tokens generated")
    eval_duration: Optional[int] = Field(default=None, description="Decoding duration in nanoseconds")
    context: Optional[List[int]] = Field(default=None, description="Output token IDs for conversation continuity")


class OllamaChatRequest(BaseModel):
    """Payload contract for Ollama POST /api/chat endpoint."""

    model: str = Field(description="Target model identifier or path")
    messages: List[OllamaChatMessage] = Field(description="Conversation history message array")
    stream: bool = Field(default=True, description="Enable streaming NDJSON responses")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Runtime generation parameters")
    adapters: Optional[List[Dict[str, Any]]] = Field(default=None, description="Active LoRA adapters to compose")
    adapter: Optional[str] = Field(default=None, description="Single LoRA adapter name or path shortcut")


class OllamaChatResponse(BaseModel):
    """Response contract for Ollama POST /api/chat endpoint."""

    model: str = Field(description="Model identifier evaluated")
    created_at: str = Field(description="ISO-8601 UTC timestamp")
    message: OllamaChatMessage = Field(description="Generated assistant message chunk or full reply")
    done: bool = Field(description="Completion status flag")
    total_duration: Optional[int] = Field(default=None, description="Total duration in nanoseconds")
    load_duration: Optional[int] = Field(default=None, description="Load duration in nanoseconds")
    prompt_eval_count: Optional[int] = Field(default=None, description="Prompt token count")
    prompt_eval_duration: Optional[int] = Field(default=None, description="Prefill duration in nanoseconds")
    eval_count: Optional[int] = Field(default=None, description="Generated token count")
    eval_duration: Optional[int] = Field(default=None, description="Decode duration in nanoseconds")


class OllamaModelDetails(BaseModel):
    """Architectural metadata for Ollama models."""

    parent_model: str = Field(default="", description="Base parent model name")
    format: str = Field(default="openvino", description="Model representation format")
    family: str = Field(default="llama", description="Model architecture family")
    families: Optional[List[str]] = Field(default=None, description="Associated architecture families")
    parameter_size: str = Field(default="3B", description="Model parameter scale estimate")
    quantization_level: str = Field(default="INT4", description="Precision quantization profile")


class OllamaModelItem(BaseModel):
    """Model summary descriptor for Ollama GET /api/tags."""

    name: str = Field(description="Model display name")
    model: str = Field(description="Model lookup identifier")
    modified_at: str = Field(description="ISO-8601 modification timestamp")
    size: int = Field(description="Total model file size in bytes")
    digest: str = Field(description="SHA256 model digest hash")
    details: OllamaModelDetails = Field(description="Architectural parameters")


class OllamaTagsResponse(BaseModel):
    """Response contract for Ollama GET /api/tags."""

    models: List[OllamaModelItem] = Field(default_factory=list, description="Available local models")


class OllamaShowRequest(BaseModel):
    """Payload contract for Ollama POST /api/show endpoint."""

    name: Optional[str] = Field(default=None, description="Target model name")
    model: Optional[str] = Field(default=None, description="Target model lookup identifier")


class OllamaShowResponse(BaseModel):
    """Response contract for Ollama POST /api/show."""

    modelfile: str = Field(default="", description="Modelfile definition contents")
    parameters: str = Field(default="", description="Parameter flags string")
    template: str = Field(default="", description="Prompt template string")
    details: OllamaModelDetails = Field(description="Model hardware and precision details")


class OllamaVersionResponse(BaseModel):
    """Response contract for Ollama GET /api/version."""

    version: str = Field(description="Server version string")


class OpenAIChatMessage(BaseModel):
    """Message item in OpenAI chat completion conversations."""

    role: str = Field(description="Participant role: system, user, or assistant")
    content: str = Field(description="Text content of message")
    name: Optional[str] = Field(default=None, description="Optional participant name")


class OpenAIChatCompletionRequest(BaseModel):
    """Payload contract for OpenAI POST /v1/chat/completions endpoint."""

    model: str = Field(description="Model identifier to use")
    messages: List[OpenAIChatMessage] = Field(description="Conversation history list")
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling probability")
    n: Optional[int] = Field(default=1, ge=1, le=1, description="Number of completions per prompt")
    stream: Optional[bool] = Field(default=False, description="Stream Server-Sent Events chunks")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequence strings")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    user: Optional[str] = Field(default=None, description="End-user identifier")


class OpenAIChatCompletionChoice(BaseModel):
    """Single completion item for non-streaming OpenAI response."""

    index: int = Field(default=0, description="Choice index")
    message: OpenAIChatMessage = Field(description="Generated assistant message")
    finish_reason: Optional[str] = Field(default="stop", description="Termination cause")


class OpenAIChatCompletionResponse(BaseModel):
    """Response contract for non-streaming OpenAI POST /v1/chat/completions."""

    id: str = Field(description="Unique completion identifier")
    object: str = Field(default="chat.completion", description="Object type string")
    created: int = Field(description="Unix epoch creation timestamp in seconds")
    model: str = Field(description="Model used for generation")
    choices: List[OpenAIChatCompletionChoice] = Field(description="Completion choices array")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Engine-reported token counts; absent when the engine did not measure them")


class OpenAIChunkDelta(BaseModel):
    """Incremental delta content in OpenAI streaming chunk."""

    role: Optional[str] = Field(default=None, description="Role identifier for first chunk")
    content: Optional[str] = Field(default=None, description="Text chunk delta")


class OpenAIChunkChoice(BaseModel):
    """Choice item inside streaming OpenAI chunk."""

    index: int = Field(default=0, description="Choice index")
    delta: OpenAIChunkDelta = Field(description="Message delta container")
    finish_reason: Optional[str] = Field(default=None, description="Termination cause")


class OpenAIChatCompletionChunk(BaseModel):
    """Single Server-Sent Event object for OpenAI streaming chat."""

    id: str = Field(description="Unique completion identifier")
    object: str = Field(default="chat.completion.chunk", description="Chunk object type")
    created: int = Field(description="Unix epoch timestamp in seconds")
    model: str = Field(description="Active model identifier")
    choices: List[OpenAIChunkChoice] = Field(description="Chunk choice list")


class OpenAIModelCard(BaseModel):
    """Model metadata item in OpenAI GET /v1/models."""

    id: str = Field(description="Model identifier")
    object: str = Field(default="model", description="Object type string")
    created: int = Field(description="Unix epoch creation timestamp")
    owned_by: str = Field(default="intel-arc-models-runner", description="Model provider entity")


class OpenAIModelListResponse(BaseModel):
    """Response contract for OpenAI GET /v1/models."""

    object: str = Field(default="list", description="Object list identifier")
    data: List[OpenAIModelCard] = Field(default_factory=list, description="Model collection list")


class OllamaPullRequest(BaseModel):
    """Payload contract for Ollama POST /api/pull endpoint."""

    name: str = Field(description="Model alias or repository identifier to download")
    insecure: bool = Field(default=False, description="Allow insecure connections")
    stream: bool = Field(default=True, description="Enable streaming NDJSON progress output")


class OllamaPullProgressResponse(BaseModel):
    """Progress frame emitted during Ollama POST /api/pull execution."""

    status: str = Field(description="Current download status")
    digest: Optional[str] = Field(default=None, description="Model short hash digest")
    total: Optional[int] = Field(default=None, description="Total bytes expected")
    completed: Optional[int] = Field(default=None, description="Bytes completed so far")


class OllamaDeleteRequest(BaseModel):
    """Payload contract for Ollama DELETE /api/delete endpoint."""

    name: Optional[str] = Field(default=None, description="Model identifier to delete")
    model: Optional[str] = Field(default=None, description="Alternate model identifier field")


class OllamaCopyRequest(BaseModel):
    """Payload contract for Ollama POST /api/copy endpoint."""

    source: str = Field(description="Existing source model alias")
    destination: str = Field(description="New destination model alias")


class OllamaStatusResponse(BaseModel):
    """Generic status response for Ollama commands."""

    status: str = Field(default="success", description="Operation status string")


class OllamaEmbeddingRequest(BaseModel):
    """Payload contract for Ollama POST /api/embeddings legacy endpoint."""

    model: str = Field(description="Target embedding model identifier")
    prompt: str = Field(description="Text string to generate embedding for")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Optional runtime parameters")
    keep_alive: Optional[str] = Field(default=None, description="Model unload timeout duration")


class OllamaEmbeddingResponse(BaseModel):
    """Response contract for Ollama POST /api/embeddings legacy endpoint."""

    embedding: List[float] = Field(description="Generated dense vector embedding")


class OllamaEmbedRequest(BaseModel):
    """Payload contract for Ollama POST /api/embed v2 multi-prompt endpoint."""

    model: str = Field(description="Target embedding model identifier")
    input: Union[str, List[str]] = Field(description="Single text string or array of strings")
    truncate: Optional[bool] = Field(default=True, description="Truncate inputs exceeding context window")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Optional runtime parameters")
    keep_alive: Optional[str] = Field(default=None, description="Model unload timeout duration")


class OllamaEmbedResponse(BaseModel):
    """Response contract for Ollama POST /api/embed v2 multi-prompt endpoint."""

    model: str = Field(description="Active embedding model evaluated")
    embeddings: List[List[float]] = Field(description="Batch of generated dense vector embeddings")
    total_duration: Optional[int] = Field(default=None, description="Total execution duration in nanoseconds")
    load_duration: Optional[int] = Field(default=None, description="Model load duration in nanoseconds")
    prompt_eval_count: Optional[int] = Field(default=None, description="Tokens evaluated")


class OpenAIEmbeddingRequest(BaseModel):
    """Payload contract for OpenAI POST /v1/embeddings endpoint."""

    model: str = Field(description="Target model identifier")
    input: Union[str, List[str]] = Field(description="Input text or array of input texts")
    user: Optional[str] = Field(default=None, description="Unique identifier representing end-user")
    dimensions: Optional[int] = Field(default=None, description="Expected dimensionality")
    encoding_format: Optional[str] = Field(default="float", description="Output encoding format")


class OpenAIEmbeddingData(BaseModel):
    """Single embedding result object in OpenAI embedding response."""

    object: str = Field(default="embedding", description="Object type identifier")
    index: int = Field(description="Index in the batch sequence")
    embedding: List[float] = Field(description="Dense vector embedding")


class OpenAIEmbeddingUsage(BaseModel):
    """Token usage metadata for embedding generation."""

    prompt_tokens: int = Field(default=0, description="Tokens evaluated in prompt inputs, as reported by the engine")
    total_tokens: int = Field(default=0, description="Total tokens consumed, as reported by the engine")


class OpenAIEmbeddingResponse(BaseModel):
    """Response contract for OpenAI POST /v1/embeddings endpoint."""

    object: str = Field(default="list", description="List object type")
    data: List[OpenAIEmbeddingData] = Field(description="Embedding data list")
    model: str = Field(description="Target model identifier evaluated")
    usage: Optional[OpenAIEmbeddingUsage] = Field(default=None, description="Usage statistics; absent when the engine did not report token counts")


class LoRASpecSchema(BaseModel):
    """Specification of an adapter binding with scale factor."""

    name: str = Field(..., description="Name or identifier of LoRA adapter")
    scale: float = Field(default=1.0, ge=0.0, le=2.0, description="Linear scaling weight lambda")
    path: Optional[str] = Field(default=None, description="Optional explicit directory path")


class LoRAAdapterResponse(BaseModel):
    """Metadata response for a discovered or loaded LoRA adapter."""

    adapter_id: str = Field(description="Unique identifier of adapter")
    adapter_path: str = Field(description="Filesystem path of adapter directory or container")
    base_model_name: str = Field(description="Base model architecture or identifier")
    rank: int = Field(description="Low-rank dimension r")
    alpha: float = Field(description="LoRA scaling parameter alpha")
    target_modules: List[str] = Field(description="Target projection module names")
    file_size_bytes: int = Field(description="Weight container file size in bytes")
    is_cached: bool = Field(description="True if currently cached in USM/host memory")
    name: Optional[str] = Field(default=None, description="Convenience alias for adapter_id")

    @model_validator(mode="after")
    def populate_name(self) -> "LoRAAdapterResponse":
        """Populate name from adapter_id."""
        if not self.name:
            self.name = self.adapter_id
        return self


class LoRALoadRequest(BaseModel):
    """Request payload to load an adapter into memory."""

    adapter: Optional[str] = Field(default=None, description="Name or filesystem path of adapter to load")
    name: Optional[str] = Field(default=None, description="Alias for adapter field")

    @model_validator(mode="after")
    def resolve_adapter_name(self) -> "LoRALoadRequest":
        """Resolve adapter from name or adapter field."""
        if not self.adapter and self.name:
            self.adapter = self.name
        if not self.adapter:
            raise ValueError("Either 'adapter' or 'name' must be provided.")
        return self


class LoRAUnloadRequest(BaseModel):
    """Request payload to evict an adapter from memory."""

    adapter: Optional[str] = Field(default=None, description="Name of adapter to unload")
    name: Optional[str] = Field(default=None, description="Alias for adapter field")

    @model_validator(mode="after")
    def resolve_adapter_name(self) -> "LoRAUnloadRequest":
        """Resolve adapter from name or adapter field."""
        if not self.adapter and self.name:
            self.adapter = self.name
        if not self.adapter:
            raise ValueError("Either 'adapter' or 'name' must be provided.")
        return self


class LoRAInspectRequest(BaseModel):
    """Request payload to inspect an adapter container without loading into memory."""

    path: Optional[str] = Field(default=None, description="Filesystem directory or safetensors path to inspect")
    name: Optional[str] = Field(default=None, description="Alias for path/name")

    @model_validator(mode="after")
    def resolve_path(self) -> "LoRAInspectRequest":
        """Resolve path from name or path field."""
        if not self.path and self.name:
            self.path = self.name
        if not self.path:
            raise ValueError("Either 'path' or 'name' must be provided.")
        return self


class BatchStatusResponse(BaseModel):
    """Telemetry response describing continuous batching queue and VRAM saturation."""

    running_sequences: int = Field(description="Number of sequences currently in running batch")
    waiting_sequences: int = Field(description="Number of sequences currently queued in waiting state")
    completed_requests: int = Field(description="Total sequences successfully completed since startup")
    total_tokens_produced: int = Field(description="Aggregate token count decoded across all sequences")
    used_kv_blocks: int = Field(description="Physical memory blocks currently allocated to sequences")
    free_kv_blocks: int = Field(description="Physical memory blocks available in the free pool")
    vram_block_utilization_pct: float = Field(description="Percentage of total physical blocks consumed")
    prefix_cache_hit_rate: float = Field(default=0.0, description="Empirical Radix prefix hit rate percentage")
    prefix_cache_hits: int = Field(default=0, description="Total Radix prefix cache hits")
    prefix_cache_misses: int = Field(default=0, description="Total Radix prefix cache misses")
    active_client_connections: int = Field(default=0, description="Active HTTP client streaming connections")
    running_sequence_ids: List[str] = Field(default_factory=list, description="IDs of currently running sequences")
    waiting_sequence_ids: List[str] = Field(default_factory=list, description="IDs of currently waiting sequences")


class BatchPreemptRequest(BaseModel):
    """Request payload to preemptively evict a running sequence from the batch."""

    seq_id: str = Field(..., description="Target sequence identifier to preempt")


