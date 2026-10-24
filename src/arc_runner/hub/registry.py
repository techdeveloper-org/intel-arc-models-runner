"""Curated catalog and registry mapping for official Intel-optimized OpenVINO IR models.

Provides alias resolution, architectural metadata, parameter counts, and quantization
specifications for verified models on Intel Arc GPUs and Intel AI Boost NPUs.
"""


from pydantic import BaseModel, Field


class CuratedModelInfo(BaseModel):
    """Architectural and repository descriptor for curated OpenVINO models."""

    alias: str = Field(description="Human-friendly lookup alias")
    repo_id: str = Field(description="Canonical Hugging Face repository identifier")
    parameter_size: str = Field(description="Model parameter count estimate")
    quantization: str = Field(description="Quantization precision and grouping format")
    family: str = Field(description="Model architectural family (llama, qwen, phi)")
    description: str = Field(description="Concise model summary description")
    target_device: str = Field(description="Recommended silicon target (GPU, NPU, HYBRID)")
    size_bytes: int = Field(description="Estimated on-disk weight footprint in bytes")
    context_window: int = Field(description="Maximum native context window tokens")
    model_type: str = Field(default="llm", description="Model modality: 'llm', 'embedding', 'vision', 'stt', 'tts', 'image_gen', 'video_gen'")
    sha256_checksum: str | None = Field(default=None, description="Published reference checksum")

    @property
    def model_id(self) -> str:
        """Alias property matching model_id contract."""
        return self.alias


class ModelRegistry:
    """Registry maintaining curated Intel-optimized models and alias resolution logic."""

    _MODELS: dict[str, CuratedModelInfo] = {
        "llama-3.2-3b": CuratedModelInfo(
            alias="llama-3.2-3b",
            repo_id="OpenVINO/Llama-3.2-3B-Instruct-int4-ov",
            parameter_size="3.21B",
            quantization="INT4 (Symmetric, Group 128)",
            family="llama",
            description="Llama 3.2 3B Instruct optimized with INT4 weights for Intel Arc GPUs",
            target_device="GPU",
            size_bytes=1820000000,
            context_window=131072,
        ),
        "llama-3.2-1b": CuratedModelInfo(
            alias="llama-3.2-1b",
            repo_id="OpenVINO/Llama-3.2-1B-Instruct-int4-ov",
            parameter_size="1.23B",
            quantization="INT4 (Symmetric, Group 128)",
            family="llama",
            description="Ultra-compact Llama 3.2 1B Instruct ideal for 5W Intel NPU speculative draft",
            target_device="NPU",
            size_bytes=740000000,
            context_window=131072,
        ),
        "qwen-2.5-3b": CuratedModelInfo(
            alias="qwen-2.5-3b",
            repo_id="OpenVINO/Qwen2.5-3B-Instruct-int4-ov",
            parameter_size="3.09B",
            quantization="INT4 (Symmetric, Group 128)",
            family="qwen",
            description="Qwen 2.5 3B Instruct optimized for coding and multilingual tasks on Arc GPUs",
            target_device="GPU",
            size_bytes=1910000000,
            context_window=32768,
        ),
        "qwen-2.5-1.5b": CuratedModelInfo(
            alias="qwen-2.5-1.5b",
            repo_id="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
            parameter_size="1.54B",
            quantization="INT4 (Symmetric, Group 128)",
            family="qwen",
            description="Qwen 2.5 1.5B Instruct lightweight model for Intel NPU cooperative drafting",
            target_device="NPU",
            size_bytes=940000000,
            context_window=32768,
        ),
        "qwen-2.5-7b": CuratedModelInfo(
            alias="qwen-2.5-7b",
            repo_id="OpenVINO/Qwen2.5-7B-Instruct-int4-ov",
            parameter_size="7.61B",
            quantization="INT4 (Symmetric, Group 128)",
            family="qwen",
            description="Qwen 2.5 7B Instruct for discrete Intel Arc GPUs (A770, A750, B580)",
            target_device="GPU",
            size_bytes=4420000000,
            context_window=32768,
        ),
        "phi-3.5-mini": CuratedModelInfo(
            alias="phi-3.5-mini",
            repo_id="OpenVINO/Phi-3.5-mini-instruct-int4-ov",
            parameter_size="3.82B",
            quantization="INT4 (Symmetric, Group 128)",
            family="phi",
            description="Microsoft Phi 3.5 mini with 128k context support on Intel Arc silicon",
            target_device="GPU",
            size_bytes=2210000000,
            context_window=131072,
        ),
        "deepseek-r1-1.5b": CuratedModelInfo(
            alias="deepseek-r1-1.5b",
            repo_id="OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov",
            parameter_size="1.77B",
            quantization="INT4 (Symmetric, Group 128)",
            family="qwen",
            description="DeepSeek R1 reasoning model distilled onto Qwen 1.5B architecture",
            target_device="HYBRID",
            size_bytes=1120000000,
            context_window=32768,
        ),
        "nomic-embed-text": CuratedModelInfo(
            alias="nomic-embed-text",
            repo_id="OpenVINO/nomic-embed-text-v1.5-fp16-ov",
            parameter_size="137M",
            quantization="FP16 (OpenVINO IR)",
            family="nomic",
            description="Nomic Embed Text v1.5 high-performance dense embedding model for Intel NPU @ 5W",
            target_device="NPU",
            size_bytes=274000000,
            context_window=8192,
        ),
        "bge-m3": CuratedModelInfo(
            alias="bge-m3",
            repo_id="OpenVINO/bge-m3-fp16-ov",
            parameter_size="567M",
            quantization="FP16 (OpenVINO IR)",
            family="bge",
            description="BAAI BGE-M3 multi-lingual, multi-granularity embedding model for RAG systems",
            target_device="NPU",
            size_bytes=1140000000,
            context_window=8192,
        ),
        "all-minilm-l6-v2": CuratedModelInfo(
            alias="all-minilm-l6-v2",
            repo_id="OpenVINO/all-MiniLM-L6-v2-fp16-ov",
            parameter_size="22.7M",
            quantization="FP16 (OpenVINO IR)",
            family="minilm",
            description="Ultra-fast lightweight sentence embedding model (384 dimensions) for local RAG",
            target_device="NPU",
            size_bytes=45000000,
            context_window=512,
        ),
        "llama-3.2-11b-vision": CuratedModelInfo(
            alias="llama-3.2-11b-vision",
            repo_id="OpenVINO/Llama-3.2-11B-Vision-Instruct-int4-ov",
            parameter_size="10.6B",
            quantization="INT4 (Symmetric, Group 128)",
            family="llama-vision",
            description="Llama 3.2 11B Vision Instruct multimodal model accelerated on Arc Xe-LPG GPUs",
            target_device="GPU",
            size_bytes=6200000000,
            context_window=131072,
        ),
        "qwen2-vl-7b": CuratedModelInfo(
            alias="qwen2-vl-7b",
            repo_id="OpenVINO/Qwen2-VL-7B-Instruct-int4-ov",
            parameter_size="7.6B",
            quantization="INT4 (Symmetric, Group 128)",
            family="qwen-vl",
            description="Qwen2-VL 7B Instruct vision-language model for document and visual reasoning",
            target_device="GPU",
            size_bytes=4500000000,
            context_window=32768,
            model_type="vision",
        ),
        "whisper-tiny": CuratedModelInfo(
            alias="whisper-tiny",
            repo_id="OpenVINO/whisper-tiny-fp16-ov",
            parameter_size="39M",
            quantization="FP16 (OpenVINO IR)",
            family="whisper",
            description="Ultra-lightweight 39M Whisper speech recognition model for low-latency ASR",
            target_device="GPU",
            size_bytes=75000000,
            context_window=448,
            model_type="stt",
        ),
        "whisper-base": CuratedModelInfo(
            alias="whisper-base",
            repo_id="OpenVINO/whisper-base-fp16-ov",
            parameter_size="74M",
            quantization="FP16 (OpenVINO IR)",
            family="whisper",
            description="Standard baseline 74M parameter Whisper ASR model for general speech transcription",
            target_device="GPU",
            size_bytes=145000000,
            context_window=448,
            model_type="stt",
        ),
        "whisper-small": CuratedModelInfo(
            alias="whisper-small",
            repo_id="OpenVINO/whisper-small-fp16-ov",
            parameter_size="244M",
            quantization="FP16 (OpenVINO IR)",
            family="whisper",
            description="High-accuracy 244M parameter Whisper model for multi-speaker transcription",
            target_device="GPU",
            size_bytes=480000000,
            context_window=448,
            model_type="stt",
        ),
        "whisper-large-v3-turbo": CuratedModelInfo(
            alias="whisper-large-v3-turbo",
            repo_id="OpenVINO/whisper-large-v3-turbo-int4-ov",
            parameter_size="809M",
            quantization="INT4 (Symmetric, Group 128)",
            family="whisper",
            description="Optimized 809M Whisper Large v3 Turbo in INT4 for enterprise ASR with low latency",
            target_device="GPU",
            size_bytes=500000000,
            context_window=448,
            model_type="stt",
        ),
        "distil-whisper-large-v3": CuratedModelInfo(
            alias="distil-whisper-large-v3",
            repo_id="OpenVINO/distil-whisper-large-v3-int4-ov",
            parameter_size="756M",
            quantization="INT4 (Symmetric, Group 128)",
            family="whisper",
            description="Distilled 756M Whisper model achieving 6x faster inference with minimal WER drift",
            target_device="GPU",
            size_bytes=450000000,
            context_window=448,
            model_type="stt",
        ),
        "kokoro-82m": CuratedModelInfo(
            alias="kokoro-82m",
            repo_id="OpenVINO/Kokoro-82M-fp16-ov",
            parameter_size="82M",
            quantization="FP16 (OpenVINO IR)",
            family="kokoro",
            description="State-of-the-art 82M StyleTTS2 Kokoro TTS model with natural prosody and multi-voice support",
            target_device="GPU",
            size_bytes=160000000,
            context_window=510,
            model_type="tts",
        ),
        "kokoro-82m-int8": CuratedModelInfo(
            alias="kokoro-82m-int8",
            repo_id="OpenVINO/Kokoro-82M-int8-ov",
            parameter_size="82M",
            quantization="INT8 (Symmetric, Group 128)",
            family="kokoro",
            description="Quantized 82M Kokoro TTS in INT8 for extreme throughput and sub-100MB footprint",
            target_device="GPU",
            size_bytes=85000000,
            context_window=510,
            model_type="tts",
        ),
        "stable-diffusion-v1-5": CuratedModelInfo(
            alias="stable-diffusion-v1-5",
            repo_id="OpenVINO/stable-diffusion-v1-5-fp16-ov",
            parameter_size="860M",
            quantization="FP16 (OpenVINO IR)",
            family="diffusion",
            description="Stable Diffusion 1.5 baseline model optimized with FP16 weights for Intel Arc GPUs",
            target_device="GPU",
            size_bytes=1720000000,
            context_window=77,
            model_type="image_gen",
        ),
        "sd-1.5": CuratedModelInfo(
            alias="sd-1.5",
            repo_id="OpenVINO/stable-diffusion-v1-5-fp16-ov",
            parameter_size="860M",
            quantization="FP16 (OpenVINO IR)",
            family="diffusion",
            description="Stable Diffusion 1.5 baseline model optimized with FP16 weights for Intel Arc GPUs",
            target_device="GPU",
            size_bytes=1720000000,
            context_window=77,
            model_type="image_gen",
        ),
        "stable-diffusion-v1-5-int8": CuratedModelInfo(
            alias="stable-diffusion-v1-5-int8",
            repo_id="OpenVINO/stable-diffusion-v1-5-int8-ov",
            parameter_size="860M",
            quantization="INT8 (Symmetric NNCF)",
            family="diffusion",
            description="Quantized Stable Diffusion 1.5 in INT8 for extreme throughput on Intel Arc XMX DPAS engines",
            target_device="GPU",
            size_bytes=860000000,
            context_window=77,
            model_type="image_gen",
        ),
        "sd-1.5-int8": CuratedModelInfo(
            alias="sd-1.5-int8",
            repo_id="OpenVINO/stable-diffusion-v1-5-int8-ov",
            parameter_size="860M",
            quantization="INT8 (Symmetric NNCF)",
            family="diffusion",
            description="Quantized Stable Diffusion 1.5 in INT8 for extreme throughput on Intel Arc XMX DPAS engines",
            target_device="GPU",
            size_bytes=860000000,
            context_window=77,
            model_type="image_gen",
        ),
        "lcm-dreamshaper-v7": CuratedModelInfo(
            alias="lcm-dreamshaper-v7",
            repo_id="rupeshs/LCM-dreamshaper-v7-openvino-int8",
            parameter_size="860M",
            quantization="INT8 (Symmetric NNCF)",
            family="lcm",
            description="Latent Consistency Model DreamShaper v7 for real-time 2-to-4 step photorealistic image synthesis",
            target_device="GPU",
            size_bytes=860000000,
            context_window=77,
            model_type="image_gen",
        ),
        "lcm": CuratedModelInfo(
            alias="lcm",
            repo_id="rupeshs/LCM-dreamshaper-v7-openvino-int8",
            parameter_size="860M",
            quantization="INT8 (Symmetric NNCF)",
            family="lcm",
            description="Latent Consistency Model DreamShaper v7 for real-time 2-to-4 step photorealistic image synthesis",
            target_device="GPU",
            size_bytes=860000000,
            context_window=77,
            model_type="image_gen",
        ),
        "sd-turbo": CuratedModelInfo(
            alias="sd-turbo",
            repo_id="rupeshs/sd-turbo-openvino",
            parameter_size="860M",
            quantization="FP16 (OpenVINO IR)",
            family="diffusion",
            description="Adversarial Diffusion Distillation (ADD) single-step real-time text-to-image generator",
            target_device="GPU",
            size_bytes=1720000000,
            context_window=77,
            model_type="image_gen",
        ),
        "stable-diffusion-xl-base-1.0": CuratedModelInfo(
            alias="stable-diffusion-xl-base-1.0",
            repo_id="OpenVINO/stable-diffusion-xl-base-1.0-fp16-ov",
            parameter_size="3.5B",
            quantization="FP16 (OpenVINO IR)",
            family="sdxl",
            description="Flagship Stable Diffusion XL 1.0 base model for high-resolution 1024x1024 generation on Arc GPUs",
            target_device="GPU",
            size_bytes=6500000000,
            context_window=77,
            model_type="image_gen",
        ),
        "sdxl": CuratedModelInfo(
            alias="sdxl",
            repo_id="OpenVINO/stable-diffusion-xl-base-1.0-fp16-ov",
            parameter_size="3.5B",
            quantization="FP16 (OpenVINO IR)",
            family="sdxl",
            description="Flagship Stable Diffusion XL 1.0 base model for high-resolution 1024x1024 generation on Arc GPUs",
            target_device="GPU",
            size_bytes=6500000000,
            context_window=77,
            model_type="image_gen",
        ),
        "sdxl-turbo": CuratedModelInfo(
            alias="sdxl-turbo",
            repo_id="rupeshs/sdxl-turbo-openvino-int8",
            parameter_size="3.5B",
            quantization="INT8 (Symmetric NNCF)",
            family="sdxl",
            description="High-resolution 1024x1024 real-time SDXL Turbo distilled for 1-4 step generation on Intel Arc",
            target_device="GPU",
            size_bytes=3200000000,
            context_window=77,
            model_type="image_gen",
        ),
        "ltx-video": CuratedModelInfo(
            alias="ltx-video",
            repo_id="Lightricks/LTX-Video-openvino",
            parameter_size="2B",
            quantization="FP16 (OpenVINO IR)",
            family="ltx-video",
            description="Lightricks LTX-Video 0.9.1 real-time DiT spatial-temporal transformer for video generation on Intel Arc GPUs",
            target_device="GPU",
            size_bytes=4200000000,
            context_window=128,
            model_type="video_gen",
        ),
        "ltx-video-int8": CuratedModelInfo(
            alias="ltx-video-int8",
            repo_id="Lightricks/LTX-Video-openvino-int8",
            parameter_size="2B",
            quantization="INT8 (Symmetric NNCF)",
            family="ltx-video",
            description="NNCF INT8 quantized Lightricks LTX-Video model optimized for high FPS throughput on Arc Xe-LPG",
            target_device="GPU",
            size_bytes=2100000000,
            context_window=128,
            model_type="video_gen",
        ),
        "cogvideox-2b": CuratedModelInfo(
            alias="cogvideox-2b",
            repo_id="THUDM/CogVideoX-2b-openvino",
            parameter_size="2.2B",
            quantization="FP16 (OpenVINO IR)",
            family="cogvideox",
            description="THUDM CogVideoX-2B expert transformer with 3D causal VAE for cinematic video synthesis on Intel Arc GPUs",
            target_device="GPU",
            size_bytes=4600000000,
            context_window=226,
            model_type="video_gen",
        ),
        "animatediff-v1-5": CuratedModelInfo(
            alias="animatediff-v1-5",
            repo_id="OpenVINO/AnimateDiff-v1-5-ov",
            parameter_size="1.2B",
            quantization="FP16 (OpenVINO IR)",
            family="animatediff",
            description="AnimateDiff motion modeling adapter integrated with Stable Diffusion 1.5 for fluid animation on Arc GPUs",
            target_device="GPU",
            size_bytes=2400000000,
            context_window=77,
            model_type="video_gen",
        ),
        "svd-xt": CuratedModelInfo(
            alias="svd-xt",
            repo_id="OpenVINO/stable-video-diffusion-img2vid-xt-ov",
            parameter_size="1.5B",
            quantization="FP16 (OpenVINO IR)",
            family="svd",
            description="Stability AI Stable Video Diffusion XT 25-frame high-resolution video generator for Intel Arc GPUs",
            target_device="GPU",
            size_bytes=3200000000,
            context_window=77,
            model_type="video_gen",
        ),
    }

    @classmethod
    def resolve(cls, alias_or_repo: str) -> CuratedModelInfo:
        """Resolve a model name alias or direct Hugging Face repository path.

        Args:
            alias_or_repo: Short alias name or full organization/model repository string.

        Returns:
            CuratedModelInfo populated with target repository coordinates.
        """
        clean_key = alias_or_repo.strip().lower()
        if clean_key in cls._MODELS:
            return cls._MODELS[clean_key]

        tag_clean = clean_key.split(":")[0]
        if tag_clean in cls._MODELS:
            return cls._MODELS[tag_clean]

        family = "custom"
        m_type = "llm"
        if "llama" in clean_key:
            family = "llama"
        elif "qwen" in clean_key:
            family = "qwen"
        elif "phi" in clean_key:
            family = "phi"
        elif "whisper" in clean_key:
            family = "whisper"
            m_type = "stt"
        elif "kokoro" in clean_key or "speech" in clean_key or "tts" in clean_key:
            family = "kokoro"
            m_type = "tts"
        elif any(k in clean_key for k in ("diffusion", "sd-", "sdxl", "lcm", "turbo", "flux")):
            family = "diffusion"
            m_type = "image_gen"
        elif any(k in clean_key for k in ("video", "ltx", "cogvideo", "svd", "animatediff", "t2v")):
            family = "video"
            m_type = "video_gen"

        return CuratedModelInfo(
            alias=alias_or_repo,
            repo_id=alias_or_repo,
            parameter_size="Custom",
            quantization="OpenVINO IR",
            family=family,
            description=f"Custom OpenVINO model repository: {alias_or_repo}",
            target_device="GPU",
            size_bytes=2000000000,
            context_window=4096,
            model_type=m_type,
        )

    @classmethod
    def get_all(cls) -> dict[str, CuratedModelInfo]:
        """Retrieve copy of all registered models mapped by slug alias.

        Returns:
            Dictionary mapping alias strings to CuratedModelInfo instances.
        """
        return cls._MODELS.copy()

    @classmethod
    def list_curated(cls) -> list[CuratedModelInfo]:
        """Retrieve ordered list of all registered curated models.

        Returns:
            List of CuratedModelInfo models available for immediate pull.
        """
        return list(cls._MODELS.values())

    @classmethod
    def is_curated(cls, model_id: str) -> bool:
        """Check whether a model identifier matches a known curated alias.

        Args:
            model_id: Identifier string to evaluate.

        Returns:
            True if identifier exists in curated registry, False otherwise.
        """
        clean_key = model_id.strip().lower().split(":")[0]
        return clean_key in cls._MODELS
