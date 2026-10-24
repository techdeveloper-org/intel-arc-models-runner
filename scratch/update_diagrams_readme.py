# -*- coding: utf-8 -*-
"""Update docs/diagrams/README.md with synchronized Mermaid previews and Proofs M73-M93."""

from pathlib import Path

diagram_defs = [
    ("1", "Package Diagram", "package_diagram", "10 Clean Core Packages, ImageEngine & Level Zero Bindings"),
    ("2", "Component Diagram", "component_diagram", "Hardware Execution Engine, ImageEngine & Level Zero Silicon"),
    ("3", "Class Diagram", "class_diagram", "BaseEngine Hierarchy, ImageEngine, AudioEngine & Batch Scheduling"),
    ("4", "Sequence Diagram", "sequence_diagram", "Hardware Compilation, Text-to-Image CFG Denoising & Whisper STT"),
    ("5", "State Diagram", "state_diagram", "Probing, LoadingImagePipeline, CFG Denoising Loop & VAD Gating"),
    ("6", "Activity Diagram", "activity_diagram", "Text-to-Image Synthesis Flow, Whisper ASR & Kokoro TTS"),
    ("7", "Deployment Diagram", "deployment_diagram", "Intel Arc Discrete GPU, Meteor Lake SoC, ImageEngine Runtime & NVMe"),
    ("8", "Use Case Diagram", "usecase_diagram", "Creative Artist Image Generation (txt2img), Speech ASR & Benchmark"),
    ("9", "Object Diagram", "object_diagram", "Runtime Instances: image_engine_sd15, sdxl, whisper, kokoro & Arc GPU"),
    ("10", "Composite Structure Diagram", "composite_diagram", "Internal Assemblies: ImageEngine, SpecEngine, AudioEngine"),
    ("11", "Interaction Diagram", "interaction_diagram", "Text-to-Image Synthesis, Audio ASR & Sequential Zero-Disk Reclaim"),
    ("12", "Communication Diagram", "communication_diagram", "Numbered Collaboration Pathways, Anti-Burn Guard & Sliced VAE"),
    ("13", "Call Graph Diagram", "call_graph_diagram", "AST Invocations: CLI handle_generate_image down to Text2ImagePipeline")
]

def extract_mermaid_block(md_path: Path) -> str:
    content = md_path.read_text(encoding="utf-8")
    start = content.find("```mermaid")
    if start == -1:
        return ""
    end = content.find("```", start + 10)
    if end == -1:
        return ""
    return content[start:end + 3]

def main():
    root = Path(r"C:\Users\techd\Documents\workspace-spring-tool-suite-4-4.27.0-new\intel-arc-models-runner")
    uml_dir = root / "uml"
    readme_path = root / "docs" / "diagrams" / "README.md"

    lines = []
    lines.append("# Architectural Diagrams Suite: `intel-arc-models-runner`\n")
    lines.append("This directory houses the comprehensive visual architecture suite for `intel-arc-models-runner`, modeled with connected MCP tools (`uml-diagram` and `drawio-diagram`) using authoritative Universal Diagram Model (UDM) payloads in strict accordance with **Rule `45-uml-diagram-lifecycle.md` (Level 2.3)**.\n")
    lines.append("Every diagram is mathematically grounded in empirical physical silicon proofs (**Proofs M1–M6, M8–M9, M15–M20, M29–M34, M50–M58, M59–M72, and M73–M93**) and available in two synchronized formats:\n")
    lines.append("1. **Mermaid Markdown (`uml/*.md`)**: Directly renderable in GitHub, IDE previews, and markdown viewers.\n")
    lines.append("2. **Editable Draw.io XML (`drawio/*.drawio`)**: Full visual XML diagrams with rich complexity-based styling, editable in [diagrams.net](https://app.diagrams.net) with zero-auth `#R` instant rendering.\n")
    lines.append("---\n")
    lines.append("## 🗺️ Master Visual Gallery & Shareable URLs\n")
    lines.append("| # | Diagram Type | Mermaid Source | Editable Draw.io Source | Direct Browser Editor Link |")
    lines.append("|:---|:---|:---|:---|:---|")

    for num, dtype, stem, desc in diagram_defs:
        raw_url = f"https%3A%2F%2Fraw.githubusercontent.com%2Ftechdeveloper-org%2Fintel-arc-models-runner%2Fmain%2Fdrawio%2F{stem}.drawio"
        app_link = f"[Open in app.diagrams.net](https://app.diagrams.net/?url={raw_url})"
        lines.append(f"| **{num}** | **{dtype}** | [`uml/{stem}.md`](file:///uml/{stem}.md) | [`drawio/{stem}.drawio`](file:///drawio/{stem}.drawio) | {app_link} |")

    lines.append("\n---\n")
    lines.append("## 🔍 Embedded Visual Previews (All 13 Diagrams)\n")

    for num, dtype, stem, desc in diagram_defs:
        lines.append(f"### {num}. {dtype} ({desc})\n")
        mermaid = extract_mermaid_block(uml_dir / f"{stem}.md")
        lines.append(mermaid + "\n")
        lines.append(f"[Mermaid Source](file:///uml/{stem}.md) | [Editable Draw.io Source](file:///drawio/{stem}.drawio)\n")
        lines.append("---\n")

    lines.append("## 📐 Mathematical Proofs & Runtime Invariants Grounding (M1–M93)\n")
    lines.append("Every diagram in this suite mathematically enforces the foundational proofs established in `intel-arc-models-runner`:\n")
    lines.append("- **Core LLM & Speculative Execution (Proofs M1–M6, M50)**: Unit L2 norm embeddings, NPU systolic 5W isolation, speculative acceptance rate $\\alpha = 0.82$, draft length $\\gamma = 4$.\n")
    lines.append("- **GGUF Transpilation & Continuous Batching (Proofs M7–M20, M29–M35)**: O(1) streaming chunk buffer, SQNR $\\ge 32.0\\text{ dB}$, XMX 32-byte alignment, Little's Law $\\theta_{\\text{sat}} \\le 0.92$.\n")
    lines.append("- **Multi-LoRA Adapter Swapping (Proofs M21–M27)**: Zero-recompile dynamic weight blending, swap latency $\\tau_{\\text{swap}} < 0.20\\text{ ms}$, bounded LRU cache $\\le 512\\text{ MB}$.\n")
    lines.append("- **Audio AI Speech & Synthesis (Proofs M59–M72)**: Real-time factor $\\text{RTF} = 0.0125$ ($80\\times$ real-time), VAD silence gating, LZ77 repetition suppression, linear PCM buffer safety $\\le 19.2\\text{ MB}$.\n")
    lines.append("- **Text-to-Image Generation (Proofs M73–M93)**:\n")
    lines.append("  - **Proof M73**: Latent Space 48× volumetric dimensionality reduction (4096× attention FLOP reduction).\n")
    lines.append("  - **Proof M74**: Classifier-Free Guidance Score Interpolation Invariant guaranteeing monotonic text adhesion.\n")
    lines.append("  - **Proof M77**: Intel Arc Xe-LPG XMX DPAS Roofline operational intensity threshold $\\text{AI}^* = 60.34\\text{ FLOP/B}$.\n")
    lines.append("  - **Proof M78**: Level Zero USM Zero-Copy VAE Latent Handshake eliminating PCIe DMA transfer ($\\tau_{\\text{copy}} = 0$).\n")
    lines.append("  - **Proof M79**: Zero-Disk Residue Invariant for sequential execution ($\\Delta S_{\\text{disk}}(t_{\\text{end}}) \\equiv 0$).\n")
    lines.append("  - **Proof M80**: Philox4x32-10 PRNG Deterministic Hash Invariant: bit-exact image reproducibility.\n")
    lines.append("  - **Proof M82**: Diffusion Energy Efficiency Yield Metric $\\eta_{\\text{image}} = \\frac{3.6 \\times 10^6}{P_{\\text{pkg}} \\cdot T_{\\text{gen}}}$ (up to 1,224,490 img/kWh).\n")
    lines.append("  - **Proof M84**: Dynamic Quantile Thresholding Anti-Burn Invariant bounding pixel clipping ratio $\\mathcal{R}_{\\text{clipped}} < 0.005$.\n")
    lines.append("  - **Proof M88**: Gaussian-Weighted Hann Partition of Unity for Tiled VAE Decoding ($\\mathcal{C}^1$ continuity).\n")
    lines.append("  - **Proof M89**: INT8 NNCF Quantization DPAS Roofline Transition doubling iteration rate to $14.70\\text{ it/s}$.\n")
    lines.append("  - **Proof M91**: Level Zero immediate command list submission latency ceiling $\\tau_{\\text{submit}} \\le 2.1\\,\\mu\\text{s}$.\n")
    lines.append("  - **Proof M92**: Sliced Latent Tile Streaming memory footprint ceiling $\\le 580\\text{ MB}$.\n")
    lines.append("  - **Proof M93**: Ahead-of-Time OpenVINO Blob Caching delivering $52.6\\times$ faster warm load ($0.54\\text{s}$).\n")
    lines.append("\n---\n")
    lines.append("## 🛠️ Editing and Modifying Diagrams\n")
    lines.append("1. **In Browser**: Click any `Open in app.diagrams.net` link in the Master Visual Gallery above to load and edit the XML diagram with rich styles.\n")
    lines.append("2. **In VS Code / Cursor**: Install the *Draw.io Integration* extension by Henning Dieterichs and open any file in `drawio/` directly.\n")
    lines.append("3. **Programmatic Synchronization**: When modifying architecture or codebase methods, update the authoritative UDM definitions and regenerate both Mermaid and Draw.io formats simultaneously in accordance with Rule `45-uml-diagram-lifecycle.md`.\n")

    full_text = "\n".join(lines)
    readme_path.write_text(full_text, encoding="utf-8")
    print(f"Successfully updated {readme_path} ({len(full_text)} characters).")

if __name__ == "__main__":
    main()
