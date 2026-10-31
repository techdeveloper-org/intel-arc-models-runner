"""Asynchronous Media and Modality API endpoints for Arc Runner.

Provides REST and streaming interfaces for Speech-to-Text (Whisper),
Text-to-Speech (Kokoro), Text-to-Image (Stable Diffusion / GenAI),
and Text-to-Video (LTX-Video / GenAI).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
import tempfile
import time
from typing import Any, Optional

import anyio
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from arc_runner.core.logging import get_logger
from arc_runner.server.errors import hidden_error

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["Media Modalities"])


class TranscribeRequest(BaseModel):
    """Payload for speech-to-text audio transcription."""

    audio_path: Optional[str] = None
    audio_base64: Optional[str] = None
    model_id: str = "whisper-base"
    device: str = "GPU"
    language: Optional[str] = None


class TTSRequest(BaseModel):
    """Payload for text-to-speech voice synthesis."""

    prompt: str = Field(..., min_length=1)
    model_id: str = "kokoro-82m"
    voice: str = "af_heart"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    device: str = "GPU"


class GenerateImageRequest(BaseModel):
    """Payload for text-to-image synthesis."""

    prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = None
    model_id: str = "sd-1.5"
    device: str = "GPU"
    steps: int = Field(20, ge=1, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)
    seed: Optional[int] = None


class GenerateVideoRequest(BaseModel):
    """Payload for text-to-video motion synthesis."""

    prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = None
    model_id: str = "ltx-video"
    device: str = "GPU"
    frames: int = Field(16, ge=8, le=128)
    fps: int = Field(16, ge=8, le=60)
    steps: int = Field(25, ge=1, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    width: int = Field(512, ge=64, le=1024)
    height: int = Field(512, ge=64, le=1024)
    seed: Optional[int] = None


@router.post("/audio/transcribe")
async def transcribe_audio(payload: TranscribeRequest, request: Request) -> dict[str, Any]:
    """Transcribe speech audio file or base64 stream using Whisper on Intel Arc GPU/CPU.

    Args:
        payload: TranscribeRequest with audio file path or base64 data.
        request: FastAPI HTTP request instance.

    Returns:
        dict: Transcription text, duration, latency, and real-time factor (RTF).
    """
    model_mgr = request.app.state.model_manager
    t_start = time.perf_counter()

    tmp_file: Optional[Path] = None
    try:
        if payload.audio_base64:
            decoded = base64.b64decode(payload.audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(decoded)
                tmp_file = Path(f.name)
            audio_path = tmp_file
        elif payload.audio_path:
            audio_path = Path(payload.audio_path).resolve()
            if not audio_path.is_file():
                raise HTTPException(status_code=404, detail=f"Audio file not found: {payload.audio_path}")
        else:
            raise HTTPException(status_code=400, detail="Either audio_path or audio_base64 must be provided.")

        from arc_runner.engine.audio_engine import AudioEngine, AudioEngineConfig

        try:
            resolved_model = model_mgr.resolve_model_path(payload.model_id) if hasattr(model_mgr, "resolve_model_path") else Path(payload.model_id)
        except Exception:
            resolved_model = Path(payload.model_id).resolve()
        cfg = AudioEngineConfig(device=payload.device, language=payload.language)
        engine = AudioEngine(cfg)

        def _run_transcribe() -> Any:
            engine.load_model(resolved_model, device=payload.device)
            return engine.transcribe(audio_path, language=payload.language)

        res = await anyio.to_thread.run_sync(_run_transcribe)
        latency_s = time.perf_counter() - t_start

        return {
            "text": res.text,
            "language": res.language,
            "audio_duration_s": res.audio_duration_s,
            "latency_s": latency_s,
            "rtf": res.rtf,
            "device": payload.device,
            "model_id": payload.model_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise hidden_error("audio transcription failed", exc, "Audio transcription failed", model=payload.model_id) from exc
    finally:
        if tmp_file and tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass


@router.post("/audio/tts")
async def synthesize_speech(payload: TTSRequest, request: Request) -> Response:
    """Synthesize high-fidelity speech waveform from text prompt using Kokoro on Intel silicon.

    Args:
        payload: TTSRequest specifying prompt, voice character, and speed.
        request: FastAPI HTTP request instance.

    Returns:
        Response: Streaming binary WAV audio with telemetry headers.
    """
    model_mgr = request.app.state.model_manager
    t_start = time.perf_counter()

    try:
        from arc_runner.engine.audio_engine import AudioEngine, AudioEngineConfig

        try:
            resolved_model = model_mgr.resolve_model_path(payload.model_id) if hasattr(model_mgr, "resolve_model_path") else Path(payload.model_id)
        except Exception:
            resolved_model = Path(payload.model_id).resolve()
        cfg = AudioEngineConfig(device=payload.device, default_voice=payload.voice)
        engine = AudioEngine(cfg)

        def _run_tts() -> tuple[bytes, float, int]:
            engine.load_model(resolved_model, device=payload.device)
            res = engine.synthesize(payload.prompt, voice=payload.voice, speed=payload.speed)
            buf = io.BytesIO()
            engine.save_audio_wav(res.audio_samples, buf, sample_rate=res.sample_rate)
            return buf.getvalue(), res.duration_s, res.sample_rate

        wav_bytes, duration_s, sample_rate = await anyio.to_thread.run_sync(_run_tts)
        latency_s = time.perf_counter() - t_start
        rtf = latency_s / max(0.001, duration_s)

        headers = {
            "Content-Type": "audio/wav",
            "Content-Disposition": f'inline; filename="synthesized_{payload.voice}.wav"',
            "X-Audio-Duration": f"{duration_s:.3f}",
            "X-Latency-Sec": f"{latency_s:.3f}",
            "X-RTF": f"{rtf:.4f}",
            "X-Sample-Rate": str(sample_rate),
        }
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)
    except Exception as exc:
        raise hidden_error("speech synthesis failed", exc, "Speech synthesis failed", model=payload.model_id) from exc


@router.post("/vision/generate-image")
async def generate_image(payload: GenerateImageRequest, request: Request) -> Response:
    """Generate image from text prompt using OpenVINO GenAI Text2Image pipeline.

    Args:
        payload: GenerateImageRequest specifying prompt, resolution, and guidance.
        request: FastAPI HTTP request instance.

    Returns:
        Response: Binary PNG image with step rate and raster throughput telemetry headers.
    """
    model_mgr = request.app.state.model_manager
    t_start = time.perf_counter()

    try:
        from arc_runner.engine.image_engine import (
            ImageEngine,
            ImageEngineConfig,
            ImageGenerationRequest,
        )

        try:
            resolved_model = model_mgr.resolve_model_path(payload.model_id) if hasattr(model_mgr, "resolve_model_path") else Path(payload.model_id)
        except Exception:
            resolved_model = Path(payload.model_id).resolve()
        cfg = ImageEngineConfig(device=payload.device, default_steps=payload.steps, default_guidance_scale=payload.guidance_scale)
        engine = ImageEngine(cfg)

        def _run_gen() -> tuple[bytes, Any]:
            engine.load_model(resolved_model, device=payload.device)
            gen_req = ImageGenerationRequest(
                prompt=payload.prompt,
                negative_prompt=payload.negative_prompt,
                num_inference_steps=payload.steps,
                guidance_scale=payload.guidance_scale,
                width=payload.width,
                height=payload.height,
                rng_seed=payload.seed,
            )
            res = engine.generate(gen_req)
            return res.image_bytes, res

        image_bytes, result_obj = await anyio.to_thread.run_sync(_run_gen)
        latency_s = time.perf_counter() - t_start

        it_rate = getattr(result_obj.breakdown, "iteration_rate_it_s", None)
        mp_s = getattr(result_obj.breakdown, "megapixels_per_sec", 0.0)

        headers = {
            "Content-Type": "image/png",
            "Content-Disposition": 'inline; filename="generated_image.png"',
            "X-Latency-Sec": f"{latency_s:.3f}",
            "X-Step-Rate": f"{it_rate:.2f}" if it_rate is not None else "n/a",
            "X-Megapixels-Per-Sec": f"{mp_s:.3f}",
            "X-Width": str(payload.width),
            "X-Height": str(payload.height),
        }
        return Response(content=image_bytes, media_type="image/png", headers=headers)
    except Exception as exc:
        raise hidden_error("image generation failed", exc, "Image generation failed", model=payload.model_id) from exc


@router.post("/vision/generate-video")
async def generate_video(payload: GenerateVideoRequest, request: Request) -> Response:
    """Generate video sequence from prompt using OpenVINO GenAI Text2Video pipeline.

    Args:
        payload: GenerateVideoRequest specifying prompt, frame count, and FPS.
        request: FastAPI HTTP request instance.

    Returns:
        Response: Binary MP4 video stream with telemetry headers.
    """
    model_mgr = request.app.state.model_manager
    t_start = time.perf_counter()

    try:
        from arc_runner.engine.video_engine import (
            VideoEngine,
            VideoEngineConfig,
            VideoGenerationRequest,
        )

        try:
            resolved_model = model_mgr.resolve_model_path(payload.model_id) if hasattr(model_mgr, "resolve_model_path") else Path(payload.model_id)
        except Exception:
            resolved_model = Path(payload.model_id).resolve()
        cfg = VideoEngineConfig(device=payload.device, default_steps=payload.steps, default_guidance_scale=payload.guidance_scale)
        engine = VideoEngine(cfg)

        def _run_video() -> tuple[bytes, Any]:
            engine.load_model(resolved_model, device=payload.device)
            v_req = VideoGenerationRequest(
                prompt=payload.prompt,
                negative_prompt=payload.negative_prompt,
                num_frames=payload.frames,
                fps=payload.fps,
                num_inference_steps=payload.steps,
                guidance_scale=payload.guidance_scale,
                width=payload.width,
                height=payload.height,
                rng_seed=payload.seed,
            )
            res = engine.generate(v_req)
            return res.video_bytes, res

        video_bytes, result_obj = await anyio.to_thread.run_sync(_run_video)
        latency_s = time.perf_counter() - t_start

        headers = {
            "Content-Type": "video/mp4",
            "Content-Disposition": 'inline; filename="generated_video.mp4"',
            "X-Latency-Sec": f"{latency_s:.3f}",
            "X-Frames": str(payload.frames),
            "X-FPS": str(payload.fps),
        }
        return Response(content=video_bytes, media_type="video/mp4", headers=headers)
    except Exception as exc:
        raise hidden_error("video generation failed", exc, "Video generation failed", model=payload.model_id) from exc
