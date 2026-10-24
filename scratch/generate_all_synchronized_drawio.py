# -*- coding: utf-8 -*-
"""Generate all 13 synchronized Draw.io XML diagrams conforming to UDM v1."""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Add claude-workflow-engine to path
sys.path.insert(0, r"C:\Users\techd\Documents\workspace-spring-tool-suite-4-4.27.0-new\claude-workflow-engine")

from langgraph_engine.diagrams.drawio.drawio_converter_enriched import DrawioConverter

UDM_MODEL_DATA = {
    "class": {
        "classes": [
            {
                "name": "ImageEngine",
                "attributes": [
                    "-_pipeline: Text2ImagePipeline",
                    "-_device: str",
                    "-_config: ImageEngineConfig",
                    "-_current_model_path: Optional[Path]"
                ],
                "methods": [
                    "+load_model(model_path, device, compile_cache)",
                    "+generate(request: ImageGenerationRequest)",
                    "-_execute_generation(request)",
                    "-_extract_pil_image(ov_tensor)",
                    "+unload()"
                ]
            },
            {
                "name": "ImageEngineConfig",
                "attributes": [
                    "+device: str",
                    "+default_steps: int",
                    "+default_guidance_scale: float",
                    "+default_width: int",
                    "+default_height: int",
                    "+enable_compile_cache: bool"
                ],
                "methods": []
            },
            {
                "name": "ImageGenerationRequest",
                "attributes": [
                    "+prompt: str",
                    "+negative_prompt: str",
                    "+width: int",
                    "+height: int",
                    "+steps: int",
                    "+guidance_scale: float",
                    "+seed: Optional[int]"
                ],
                "methods": []
            },
            {
                "name": "ImageGenerationBreakdown",
                "attributes": [
                    "+text_encoder_ms: float",
                    "+denoise_unet_ms: float",
                    "+vae_decoder_ms: float"
                ],
                "methods": []
            },
            {
                "name": "ImageGenerationResult",
                "attributes": [
                    "+image: PIL.Image",
                    "+latency_s: float",
                    "+step_latency_ms: float",
                    "+iteration_rate_it_s: float",
                    "+megapixels_per_sec: float",
                    "+device_used: str",
                    "+seed_used: int"
                ],
                "methods": []
            },
            {
                "name": "BaseEngine",
                "attributes": [
                    "+is_loaded",
                    "+model_path",
                    "+device"
                ],
                "methods": [
                    "+*load_model()",
                    "+*generate_stream()",
                    "+*get_telemetry()",
                    "+*unload()"
                ]
            },
            {
                "name": "OpenVINOGenAIEngine",
                "attributes": ["-_pipeline", "-_device"],
                "methods": ["+load_model()", "+generate_stream()", "-_build_ov_generation_config()"]
            },
            {
                "name": "HybridSpeculativeEngine",
                "attributes": ["-target_engine", "-draft_engine", "+current_gamma"],
                "methods": ["+load_model()", "+generate_stream()", "-_validate_path()"]
            },
            {
                "name": "AudioEngine",
                "attributes": ["-_pipeline", "-_device", "-_config: AudioEngineConfig"],
                "methods": ["+load_model()", "+transcribe()", "+synthesize()", "+unload()"]
            },
            {
                "name": "ContinuousBatcher",
                "attributes": ["+active_sequences_count", "+waiting_sequences_count"],
                "methods": ["+add_request()", "+step()", "+preempt_sequence()"]
            }
        ],
        "relationships": [
            {"source": "ImageEngine", "target": "BaseEngine", "type": "generalization"},
            {"source": "AudioEngine", "target": "BaseEngine", "type": "generalization"},
            {"source": "OpenVINOGenAIEngine", "target": "BaseEngine", "type": "generalization"},
            {"source": "HybridSpeculativeEngine", "target": "BaseEngine", "type": "generalization"},
            {"source": "ImageEngine", "target": "ImageEngineConfig", "type": "composition"},
            {"source": "ImageEngine", "target": "ImageGenerationRequest", "type": "dependency"},
            {"source": "ImageEngine", "target": "ImageGenerationResult", "type": "dependency"}
        ]
    },

    "package": {
        "packages": [
            {"name": "arc_runner.engine", "contents": ["ImageEngine", "AudioEngine", "OpenVINOGenAIEngine", "ContinuousBatcher"]},
            {"name": "arc_runner.cli", "contents": ["main (run, transcribe, tts, generate-image)"]},
            {"name": "arc_runner.hub", "contents": ["registry (sd-1.5, sdxl, lcm)", "downloader"]},
            {"name": "arc_runner.server", "contents": ["app", "batch_pipeline", "model_manager"]},
            {"name": "arc_runner.devices", "contents": ["detector", "sysman", "models"]},
            {"name": "arc_runner.quant", "contents": ["gguf_parser", "transpiler"]},
            {"name": "arc_runner.core", "contents": ["constants", "logging"]}
        ],
        "dependencies": [
            {"source": "arc_runner.cli", "target": "arc_runner.engine", "label": "invokes"},
            {"source": "arc_runner.cli", "target": "arc_runner.server", "label": "launches"},
            {"source": "arc_runner.server", "target": "arc_runner.engine", "label": "dispatches"},
            {"source": "arc_runner.engine", "target": "arc_runner.devices", "label": "hardware_ops"},
            {"source": "arc_runner.engine", "target": "arc_runner.hub", "label": "resolves_models"}
        ]
    },

    "component": {
        "components": [
            {"name": "Image_Engine", "provides": ["generate()", "Level Zero tau <= 2.1us (Proof M91)", "Anti-Burn Bound (Proof M84)"]},
            {"name": "Audio_Engine", "provides": ["transcribe()", "synthesize()", "VAD Gating (Proof M69)"]},
            {"name": "OpenVINOGenAIEngine", "provides": ["generate_stream()", "Unified Silicon"]},
            {"name": "ContinuousBatcher", "provides": ["Little's Law theta_sat <= 0.92 (Proof M30)"]},
            {"name": "ModelManager", "provides": ["Engine Registry", "VRAM Budgeting"]},
            {"name": "CLI_Subsystem", "provides": ["CLI (run, transcribe, tts, generate-image)"]},
            {"name": "Intel_Arc_GPU", "provides": ["32 Xe-Cores / 256 XMX", "560 GB/s GDDR6"]},
            {"name": "Intel_Ultra_NPU", "provides": ["48 TOPs Systolic Array", "5W TDP (Proof M50)"]},
            {"name": "Host_CPU", "provides": ["AVX2 / AMX Matrix Host Fallback"]}
        ],
        "dependencies": [
            {"source": "CLI_Subsystem", "target": "Image_Engine", "label": "direct_generate_image"},
            {"source": "CLI_Subsystem", "target": "Audio_Engine", "label": "direct_transcribe_tts"},
            {"source": "ModelManager", "target": "Image_Engine", "label": "initialize_image"},
            {"source": "Image_Engine", "target": "Intel_Arc_GPU", "label": "zeCommandListExecuteImmediate"},
            {"source": "Image_Engine", "target": "Host_CPU", "label": "cpu_fallback"},
            {"source": "Audio_Engine", "target": "Intel_Arc_GPU", "label": "zeCommandListExecuteImmediate"}
        ]
    },

    "sequence": {
        "participants": [
            "Client",
            "ImageEngine",
            "PhiloxPRNG",
            "LevelZero_GPU",
            "AntiBurnGuard",
            "DiskReclaimer"
        ],
        "call_chains": [
            {"caller": "Client", "callee": "ImageEngine", "label": "generate(prompt, 512x512, 25 steps, cfg=7.5)"},
            {"caller": "ImageEngine", "callee": "PhiloxPRNG", "label": "init_latents(seed=42) [Proof M80]"},
            {"caller": "ImageEngine", "callee": "LevelZero_GPU", "label": "Text2ImagePipeline.generate() [Proof M91]"},
            {"caller": "LevelZero_GPU", "callee": "LevelZero_GPU", "label": "UNet CFG Denoise 14.7 it/s [Proof M74, M89]"},
            {"caller": "LevelZero_GPU", "callee": "AntiBurnGuard", "label": "Dynamic Quantile Thresholding [Proof M84]"},
            {"caller": "AntiBurnGuard", "callee": "LevelZero_GPU", "label": "Clamped Latents (R_clipped < 0.005)"},
            {"caller": "LevelZero_GPU", "callee": "ImageEngine", "label": "USM zero-copy ov.Tensor handoff [Proof M78]"},
            {"caller": "ImageEngine", "callee": "Client", "label": "ImageGenerationResult (PIL.Image, Proof M82)"},
            {"caller": "ImageEngine", "callee": "DiskReclaimer", "label": "cleanup_model_dir() [Proof M79: Delta S_disk = 0]"}
        ]
    },

    "state": {
        "states": [
            {"name": "Uninitialized"},
            {"name": "ProbingHardware"},
            {"name": "IdleReady"},
            {
                "name": "LoadingImagePipeline",
                "is_composite": True,
                "sub_states": [
                    "DetectSiliconTarget",
                    "LevelZeroBinding (Proof M91)",
                    "CheckAOTBlobCache (Proof M93)",
                    "PipelineCompiled"
                ]
            },
            {
                "name": "ImageGenerationExecution",
                "is_composite": True,
                "sub_states": [
                    "ValidateDimensions (Proof M73)",
                    "PhiloxPRNGLatentInit (Proof M80)",
                    "CFGDenoisingLoop (Proof M74, M75)",
                    "AntiBurnQuantileClamping (Proof M84)",
                    "SlicedVAEDecoding (Proof M92)",
                    "USMZeroCopyExtraction (Proof M78)"
                ]
            },
            {
                "name": "AcousticVADGating",
                "is_composite": True,
                "sub_states": [
                    "ComputeRMS",
                    "SpectralEntropyCheck (Proof M69)",
                    "ActiveSpeechVerified"
                ]
            }
        ],
        "transitions": [
            {"source": "Uninitialized", "target": "ProbingHardware", "label": "startup"},
            {"source": "ProbingHardware", "target": "IdleReady", "label": "hardware_detected"},
            {"source": "IdleReady", "target": "LoadingImagePipeline", "label": "load_image_model"},
            {"source": "LoadingImagePipeline", "target": "IdleReady", "label": "pipeline_active"},
            {"source": "IdleReady", "target": "ImageGenerationExecution", "label": "generate_image"},
            {"source": "ImageGenerationExecution", "target": "IdleReady", "label": "emit_image_result"},
            {"source": "IdleReady", "target": "AcousticVADGating", "label": "transcribe_request"},
            {"source": "AcousticVADGating", "target": "IdleReady", "label": "emit_transcription"}
        ]
    },

    "activity": {
        "steps": [
            {"name": "Ingest_Prompt_And_Parameters (Steps, CFG, WxH)", "type": "action"},
            {"name": "Validate_Dimensions_Multiples_Of_8 (Proof M73)", "type": "decision"},
            {"name": "Seed_Philox_PRNG_Latents (Proof M80)", "type": "action"},
            {"name": "Submit_LevelZero_UNet_CmdList (Proof M91)", "type": "action"},
            {"name": "Execute_DPAS_Roofline_Denoise_14_7_it_s (Proof M77, M89)", "type": "action"},
            {"name": "Apply_Dynamic_Quantile_AntiBurn_Clamp (Proof M84)", "type": "action"},
            {"name": "Execute_Sliced_VAE_Decode_RAM_580MB (Proof M88, M92)", "type": "action"},
            {"name": "Extract_ZeroCopy_USM_Array (Proof M78)", "type": "action"},
            {"name": "Assemble_ImageGenerationResult_DTO (Proof M82)", "type": "action"},
            {"name": "Purge_Model_Directory_Delta_S_0 (Proof M79)", "type": "action"}
        ]
    },

    "deployment": {
        "nodes": [
            {
                "name": "Host Workstation (Python 3.10-3.13)",
                "artifacts": [
                    "arc-runner Daemon (:11434)",
                    "Studio WebUI (:11434/ui)",
                    "ImageEngine Runtime (SD 1.5, SDXL, LCM, Turbo)",
                    "AudioEngine Runtime (Whisper & Kokoro)",
                    "CLI Handlers (txt2img, transcribe, tts)",
                    "openvino_genai.dll (Level Zero Loader)"
                ]
            },
            {
                "name": "Intel Arc Discrete GPU (ACM-G10 / BMG-G21)",
                "artifacts": [
                    "32 Xe-Cores / 256 XMX (260 TFLOPs)",
                    "16 GB Dedicated GDDR6 (560 GB/s)",
                    "Level Zero Immediate Cmds (tau <= 2.1us, Proof M91)",
                    "USM Zero-Copy VAE Latent Handshake (Proof M78)"
                ]
            },
            {
                "name": "Intel Core Ultra SoC (Meteor Lake / Lunar Lake)",
                "artifacts": [
                    "Host CPU (AVX2/AMX Matrix Fallback)",
                    "Intel NPU 3720/4000 (48 TOPs, 5W TDP, Proof M50)",
                    "16-64 GB LPDDR5X (136 GB/s) Unified Host Memory"
                ]
            },
            {
                "name": "NVMe SSD Storage (7000 MB/s)",
                "artifacts": [
                    "~/.arc-runner/models/ (OpenVINO IR Cache)",
                    "~/.arc-runner/cache/image/ (Zero-Disk Residue, Proof M79)",
                    "~/.arc-runner/aot_cache/ (AOT Compiled Blobs, Proof M93)"
                ]
            }
        ],
        "connections": [
            {"source": "Host Workstation (Python 3.10-3.13)", "target": "Intel Arc Discrete GPU (ACM-G10 / BMG-G21)", "label": "PCIe Gen4 x16 DMA 32 GB/s"},
            {"source": "Host Workstation (Python 3.10-3.13)", "target": "Intel Core Ultra SoC (Meteor Lake / Lunar Lake)", "label": "Direct Ring Interconnect"},
            {"source": "Host Workstation (Python 3.10-3.13)", "target": "NVMe SSD Storage (7000 MB/s)", "label": "PCIe Gen4 x4 DMA 7000 MB/s"}
        ]
    },

    "usecase": {
        "actors": [
            "Developer / CLI User",
            "Creative Artist / Prompt Engineer",
            "Ecosystem Client (SDK)",
            "Benchmark Harness"
        ],
        "use_cases": [
            "Synthesize Images from Text Prompts (SD 1.5, SDXL, LCM, Turbo, Proof M73)",
            "Classifier-Free Guidance & Negative Steering (Proof M74, M83, M87)",
            "Anti-Burn Quantile Clamping Guard (R_clipped < 0.005, Proof M84)",
            "Latent Consistency Trajectory Acceleration (N <= 4 Steps, Proof M76, M90)",
            "Memory-Bounded Sliced VAE Decoding (RAM <= 580MB, Proof M92)",
            "USM Zero-Copy Tensor Extraction (tau_copy = 0, Proof M78)",
            "Level Zero Immediate CmdList Execution (tau <= 2.1us, Proof M91)",
            "Sequential 1-by-1 Model Benchmarking (Zero-Disk Residue, Proof M79)",
            "Transcribe Speech Waveforms (Whisper, Proof M59, M69, M70)",
            "Synthesize Natural Speech (Kokoro 82M, Proof M64)"
        ],
        "system_name": "arc-runner Execution System",
        "associations": [
            {"actor": "Creative Artist / Prompt Engineer", "use_case": "Synthesize Images from Text Prompts (SD 1.5, SDXL, LCM, Turbo, Proof M73)"},
            {"actor": "Developer / CLI User", "use_case": "Synthesize Images from Text Prompts (SD 1.5, SDXL, LCM, Turbo, Proof M73)"},
            {"actor": "Developer / CLI User", "use_case": "Sequential 1-by-1 Model Benchmarking (Zero-Disk Residue, Proof M79)"},
            {"actor": "Benchmark Harness", "use_case": "Sequential 1-by-1 Model Benchmarking (Zero-Disk Residue, Proof M79)"}
        ]
    },

    "object": {
        "objects": [
            {
                "name": "image_engine_sd15",
                "class": "ImageEngine",
                "values": {
                    "model_name": "'stable-diffusion-v1-5-int8'",
                    "precision": "'INT8 NNCF'",
                    "resolution": "'512x512'",
                    "steps": "25",
                    "iteration_rate": "14.70 it/s (Proof M89)",
                    "speedup_vs_cpu": "6.10x",
                    "energy_yield": "100781 img/kWh (Proof M82)",
                    "anti_burn_clipping": "0.0 (Proof M84)",
                    "zero_disk_residue": "true (Proof M79)"
                }
            },
            {
                "name": "image_engine_sdxl",
                "class": "ImageEngine",
                "values": {
                    "model_name": "'sdxl-turbo'",
                    "precision": "'INT8 NNCF'",
                    "resolution": "'1024x1024'",
                    "steps": "1",
                    "latency_s": "4.003s",
                    "speedup_vs_cpu": "6.60x",
                    "sliced_vae_ram": "540MB (Proof M92 <= 580MB)",
                    "zero_disk_residue": "true (Proof M79)"
                }
            },
            {
                "name": "audio_engine_whisper",
                "class": "AudioEngine",
                "values": {
                    "model_name": "'whisper-large-v3-turbo'",
                    "device": "'GPU'",
                    "rtf": "0.0125 (80x RT, Proof M59)",
                    "zero_disk_residue": "true (Proof M66)"
                }
            },
            {
                "name": "audio_engine_kokoro",
                "class": "AudioEngine",
                "values": {
                    "model_name": "'kokoro-82m-int8'",
                    "device": "'GPU'",
                    "sample_rate": "24000",
                    "voice": "'af_heart'",
                    "rtf": "0.0754 (Proof M59)"
                }
            },
            {
                "name": "device_gpu",
                "class": "IntelArcGPU",
                "values": {
                    "xe_cores": "8",
                    "vram_mb": "4620",
                    "level_zero_immediate": "true (Proof M91)"
                }
            },
            {
                "name": "device_npu",
                "class": "IntelNPU",
                "values": {
                    "tops": "48",
                    "power_tdp": "5.0W (Proof M50)",
                    "mode": "'draft_accelerator'"
                }
            }
        ],
        "links": [
            {"source": "image_engine_sd15", "target": "device_gpu", "label": "dispatches"},
            {"source": "image_engine_sdxl", "target": "device_gpu", "label": "dispatches"},
            {"source": "audio_engine_whisper", "target": "device_gpu", "label": "dispatches"},
            {"source": "audio_engine_kokoro", "target": "device_gpu", "label": "dispatches"}
        ]
    },

    "composite": {
        "components": [
            {
                "name": "ImageEngine (Text-to-Image Synthesis)",
                "parts": [
                    "CLIPTextEncoder (77-Token Context, Proof M73)",
                    "PhiloxPRNGInitializer (Deterministic Noise, Proof M80)",
                    "CFGInterpolationEngine (eps_uncond + s*(eps_cond - eps_uncond), Proof M74)",
                    "UNetDenoiserDPAS (8 Xe-Core XMX Matrix Kernel, Proof M75, M77, M89)",
                    "DynamicQuantileClamper (Anti-Burn Guard, Proof M84)",
                    "VAETileDecoder (Sliced Latent Decoder <= 580MB, Proof M88, M92)",
                    "USMZeroCopyExtractor (Shared Memory Extractor, Proof M78)"
                ],
                "ports": ["InPort_Prompt", "InPort_Config", "OutPort_Image", "Port_Telemetry"]
            },
            {
                "name": "HybridSpeculativeEngine",
                "parts": [
                    "DraftController (NPU Coordinator, gamma=4, Proof M50)",
                    "TargetVerifier (Arc GPU Batch Verifier, Proof M6)",
                    "BonusTokenEvaluator (E[K]=3.36)"
                ],
                "ports": ["prompt_in", "token_out", "telemetry"]
            },
            {
                "name": "AudioEngine (Unified STT & TTS)",
                "parts": [
                    "AcousticNormalizer",
                    "VADSilenceGater (Proof M69)",
                    "WhisperASRCore (Proof M60, M61)",
                    "KokoroTTSCore (Proof M64)",
                    "LZ77RepetitionGuard (Proof M70)"
                ],
                "ports": ["audio_in", "text_in", "transcript_out", "audio_out"]
            }
        ]
    },

    "interaction": {
        "steps": [
            {"type": "init", "name": "Client Ingest Prompt & Parameters"},
            {"type": "action", "name": "Validate Dimensions Multiple of 8 (Proof M73)"},
            {"type": "action", "name": "Seed Philox4x32-10 PRNG (Proof M80)"},
            {"type": "action", "name": "Level Zero UNet CFG Denoise 14.7 it/s (Proof M74, M91)"},
            {"type": "decision", "name": "Dynamic Quantile Saturation Clamp s_t (Proof M84)"},
            {"type": "action", "name": "Sliced VAE Decoding <= 580MB RAM (Proof M88, M92)"},
            {"type": "action", "name": "Zero-Copy Pointer Handoff tau_copy=0 (Proof M78)"},
            {"type": "ref", "name": "Emit ImageGenerationResult DTO (Proof M82)"},
            {"type": "end", "name": "Purge Directory: Delta S_disk = 0 bytes (Proof M79)"}
        ]
    },

    "communication": {
        "participants": [
            "Client",
            "ImageEngine",
            "PhiloxPRNG",
            "LevelZeroDriver",
            "IntelArcGPU",
            "AntiBurnClamper",
            "SlicedVAEDecoder",
            "DiskReclaimer"
        ],
        "messages": [
            {"from": "Client", "to": "ImageEngine", "label": "1. generate(prompt, cfg=7.5, steps=25)"},
            {"from": "ImageEngine", "to": "PhiloxPRNG", "label": "2. init_latents(seed=42) [Proof M80]"},
            {"from": "ImageEngine", "to": "LevelZeroDriver", "label": "3. dispatch_denoise_loop() [Proof M91]"},
            {"from": "LevelZeroDriver", "to": "IntelArcGPU", "label": "4. zeCommandListExecuteImmediate() 14.7 it/s [Proof M77, M89]"},
            {"from": "IntelArcGPU", "to": "AntiBurnClamper", "label": "5. raw latents -> quantile thresholding [Proof M84]"},
            {"from": "AntiBurnClamper", "to": "SlicedVAEDecoder", "label": "6. clamped latents -> sliced tile decode <= 580MB [Proof M92]"},
            {"from": "SlicedVAEDecoder", "to": "ImageEngine", "label": "7. decoded RGB array via USM zero-copy handoff [Proof M78]"},
            {"from": "ImageEngine", "to": "Client", "label": "8. ImageGenerationResult (PIL.Image, 100781 img/kWh, Proof M82)"},
            {"from": "ImageEngine", "to": "DiskReclaimer", "label": "9. cleanup_model_dir() [Proof M79: Delta S_disk = 0]"}
        ]
    },

    "call_graph": {
        "methods": [
            {"fqn": "arc_runner.cli.main.handle_generate_image", "name": "handle_generate_image", "parent_class": "CLI"},
            {"fqn": "arc_runner.engine.ImageEngine.load_model", "name": "load_model", "parent_class": "ImageEngine"},
            {"fqn": "arc_runner.engine.ImageEngine.generate", "name": "generate", "parent_class": "ImageEngine"},
            {"fqn": "arc_runner.engine.ImageEngine._execute_generation", "name": "_execute_generation", "parent_class": "ImageEngine"},
            {"fqn": "arc_runner.engine.ImageEngine._extract_pil_image", "name": "_extract_pil_image", "parent_class": "ImageEngine"},
            {"fqn": "arc_runner.engine.ImageEngine.unload", "name": "unload", "parent_class": "ImageEngine"},
            {"fqn": "openvino_genai.Text2ImagePipeline.generate", "name": "generate", "parent_class": "Text2ImagePipeline"},
            {"fqn": "scratch.run_live_image_benchmarks.main", "name": "main", "parent_class": "ImageBenchmarkHarness"},
            {"fqn": "scratch.run_live_image_benchmarks.benchmark_single_model", "name": "benchmark_single_model", "parent_class": "ImageBenchmarkHarness"},
            {"fqn": "scratch.run_live_image_benchmarks.cleanup_model_dir", "name": "cleanup_model_dir", "parent_class": "ImageBenchmarkHarness"}
        ],
        "edges": [
            {"from": "arc_runner.cli.main.handle_generate_image", "to": "arc_runner.engine.ImageEngine.load_model", "label": "loads"},
            {"from": "arc_runner.cli.main.handle_generate_image", "to": "arc_runner.engine.ImageEngine.generate", "label": "invokes"},
            {"from": "arc_runner.engine.ImageEngine.generate", "to": "arc_runner.engine.ImageEngine._execute_generation", "label": "calls"},
            {"from": "arc_runner.engine.ImageEngine._execute_generation", "to": "openvino_genai.Text2ImagePipeline.generate", "label": "Level Zero dispatch"},
            {"from": "arc_runner.engine.ImageEngine._execute_generation", "to": "arc_runner.engine.ImageEngine._extract_pil_image", "label": "extracts"},
            {"from": "scratch.run_live_image_benchmarks.main", "to": "scratch.run_live_image_benchmarks.benchmark_single_model", "label": "benchmarks"},
            {"from": "scratch.run_live_image_benchmarks.benchmark_single_model", "to": "arc_runner.engine.ImageEngine.load_model", "label": "compiles"},
            {"from": "scratch.run_live_image_benchmarks.benchmark_single_model", "to": "arc_runner.engine.ImageEngine.generate", "label": "evaluates"},
            {"from": "scratch.run_live_image_benchmarks.benchmark_single_model", "to": "arc_runner.engine.ImageEngine.unload", "label": "unloads"},
            {"from": "scratch.run_live_image_benchmarks.benchmark_single_model", "to": "scratch.run_live_image_benchmarks.cleanup_model_dir", "label": "evicts (Proof M79)"}
        ]
    }
}

def main():
    converter = DrawioConverter()
    out_dir = Path(r"C:\Users\techd\Documents\workspace-spring-tool-suite-4-4.27.0-new\intel-arc-models-runner\drawio")
    out_dir.mkdir(parents=True, exist_ok=True)

    diagram_types = [
        ("class", "class_diagram.drawio"),
        ("package", "package_diagram.drawio"),
        ("component", "component_diagram.drawio"),
        ("sequence", "sequence_diagram.drawio"),
        ("state", "state_diagram.drawio"),
        ("activity", "activity_diagram.drawio"),
        ("deployment", "deployment_diagram.drawio"),
        ("usecase", "usecase_diagram.drawio"),
        ("object", "object_diagram.drawio"),
        ("composite", "composite_diagram.drawio"),
        ("interaction", "interaction_diagram.drawio"),
        ("communication", "communication_diagram.drawio"),
        ("call_graph", "call_graph_diagram.drawio"),
    ]

    print("Generating 13 Synchronized Draw.io Diagrams using UDM v1 payloads...")
    for dtype, filename in diagram_types:
        payload = UDM_MODEL_DATA.get(dtype, {})
        xml_str = converter.convert(dtype, payload)
        
        # Verify valid XML
        ET.fromstring(xml_str)
        
        target_path = out_dir / filename
        target_path.write_text(xml_str, encoding="utf-8")
        cell_count = len(ET.fromstring(xml_str).findall(".//mxCell"))
        print(f"  [OK] {filename:<28} ({len(xml_str):>5} bytes, {cell_count:>3} cells)")

    print("\nAll 13 Draw.io XML files successfully generated and verified!")

if __name__ == "__main__":
    main()
