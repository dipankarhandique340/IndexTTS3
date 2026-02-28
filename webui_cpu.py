#!/usr/bin/env python3
"""
IndexTTS2 — CPU/Low-VRAM Gradio Web Interface
================================================
A Gradio web UI that uses the CPU-optimized inference engine.

Run:
  .venv/bin/python webui_cpu.py
  .venv/bin/python webui_cpu.py --port 7860
"""

import json
import logging
import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

os.environ.setdefault("INDEXTTS_USE_DEEPSPEED", "0")
os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import torch

# Import our CPU-optimized engine
from run_cpu import IndexTTS2CPU, download_checkpoints, print_banner

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="IndexTTS2 CPU WebUI")
parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
parser.add_argument("--model-dir", type=str, default="checkpoints", help="Model dir")
parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda:0)")
parser.add_argument("--no-quantize", action="store_true", help="Disable INT8 quantization")
parser.add_argument("--share", action="store_true", help="Create public Gradio link")
cmd_args = parser.parse_args()

# ---------------------------------------------------------------------------
# Download & Load Model
# ---------------------------------------------------------------------------
print_banner("IndexTTS2 CPU WebUI — Starting")

# download_checkpoints(cmd_args.model_dir)

print_banner("Loading Models (this may take 1-2 minutes)")
tts = IndexTTS2CPU(
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    model_dir=cmd_args.model_dir,
    device=cmd_args.device,
    quantize=not cmd_args.no_quantize,
    verbose=True,
)

logger = logging.getLogger(__name__)
os.makedirs("outputs", exist_ok=True)

# Emotion choices
EMO_CHOICES = [
    "Match prompt audio",
    "Use emotion reference audio",
    "Use emotion vector",
    "Use emotion text description",
]

# Load example cases
example_cases = []
cases_file = os.path.join("examples", "cases.jsonl")
if os.path.exists(cases_file):
    with open(cases_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                emo_audio = os.path.join("examples", ex["emo_audio"]) if ex.get("emo_audio") else None
                example_cases.append([
                    os.path.join("examples", ex.get("prompt_audio", "voice_01.wav")),
                    EMO_CHOICES[ex.get("emo_mode", 0)],
                    ex.get("text", ""),
                    emo_audio,
                    ex.get("emo_weight", 1.0),
                    ex.get("emo_text", ""),
                    ex.get("emo_vec_1", 0), ex.get("emo_vec_2", 0),
                    ex.get("emo_vec_3", 0), ex.get("emo_vec_4", 0),
                    ex.get("emo_vec_5", 0), ex.get("emo_vec_6", 0),
                    ex.get("emo_vec_7", 0), ex.get("emo_vec_8", 0),
                ])
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Generation function
# ---------------------------------------------------------------------------
def generate_speech(
    prompt_audio, text, emo_mode,
    emo_ref_audio, emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text, emo_random,
    max_tokens_per_sentence,
    do_sample, top_p, top_k, temperature,
    length_penalty, num_beams, repetition_penalty, max_mel_tokens,
    progress=gr.Progress()
):
    if not prompt_audio:
        gr.Warning("Please upload a prompt audio file first!")
        return None
    if not text or not text.strip():
        gr.Warning("Please enter text to synthesize!")
        return None

    output_path = os.path.join("outputs", f"gen_{int(time.time())}.wav")
    tts.gr_progress = progress

    # Parse emotion mode
    if isinstance(emo_mode, str):
        emo_mode_idx = EMO_CHOICES.index(emo_mode) if emo_mode in EMO_CHOICES else 0
    else:
        emo_mode_idx = emo_mode or 0

    # Build generation kwargs
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": int(num_beams),
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }

    # Emotion settings
    emo_audio = None
    emo_vec = None
    use_emo_text = False

    if emo_mode_idx == 0:
        # Match prompt audio
        emo_audio = None
        emo_weight = 1.0
    elif emo_mode_idx == 1:
        # Use emotion reference audio
        emo_audio = emo_ref_audio
    elif emo_mode_idx == 2:
        # Use emotion vector
        emo_vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec_sum = sum(emo_vec)
        if vec_sum > 1.5:
            gr.Warning("Emotion vector sum cannot exceed 1.5!")
            return None
    elif emo_mode_idx == 3:
        # Use emotion text description
        use_emo_text = True

    try:
        result = tts.infer(
            spk_audio_prompt=prompt_audio,
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_audio,
            emo_alpha=float(emo_weight),
            emo_vector=emo_vec,
            use_emo_text=use_emo_text,
            emo_text=emo_text if use_emo_text else None,
            use_random=emo_random,
            verbose=True,
            max_text_tokens_per_segment=int(max_tokens_per_sentence),
            **kwargs
        )
        if result and os.path.exists(result):
            return result
        else:
            gr.Warning("Generation failed — no output produced.")
            return None
    except Exception as e:
        logger.exception("Generation failed")
        gr.Warning(f"Error: {e}")
        return None


def preview_sentences(text, max_tokens):
    if not text or not text.strip():
        return []
    tokens = tts.tokenizer.tokenize(text)
    segments = tts.tokenizer.split_segments(tokens, int(max_tokens))
    return [[i+1, "".join(s), len(s)] for i, s in enumerate(segments)]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
css = """
.main-title { text-align: center; margin-bottom: 0.5em; }
.subtitle { text-align: center; color: #888; font-size: 0.95em; margin-top: 0; }
.generate-btn { min-height: 45px !important; font-size: 1.1em !important; }
"""

with gr.Blocks(
    title="IndexTTS2 — CPU Edition",
    css=css,
    theme=gr.themes.Soft()
) as demo:

    gr.HTML("""
    <h1 class="main-title">🎙️ IndexTTS2 — CPU/Low-VRAM Edition</h1>
    <p class="subtitle">
        Emotionally Expressive Zero-Shot Text-to-Speech · Running on CPU with INT8 Quantization
    </p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt_audio = gr.Audio(
                label="🎤 Speaker Reference Audio",
                type="filepath",
                sources=["upload", "microphone"]
            )

        with gr.Column(scale=2):
            input_text = gr.TextArea(
                label="📝 Text to Synthesize",
                placeholder="Type or paste your text here...\nSupports English, Chinese, and mixed text.",
                lines=5,
            )

    gen_button = gr.Button(
        "🔊 Generate Speech",
        variant="primary",
        elem_classes=["generate-btn"]
    )
    output_audio = gr.Audio(label="🎵 Generated Speech", type="filepath")

    # Emotion Settings
    with gr.Accordion("🎭 Emotion Settings", open=False):
        emo_mode = gr.Radio(
            choices=EMO_CHOICES,
            type="index",
            value=EMO_CHOICES[0],
            label="Emotion Control Mode",
        )

        with gr.Group(visible=False) as emo_ref_group:
            emo_ref_audio = gr.Audio(label="Emotion Reference Audio", type="filepath")
            emo_weight = gr.Slider(label="Emotion Weight", minimum=0.0, maximum=1.6, value=0.8, step=0.01)

        emo_random = gr.Checkbox(label="Random Emotion Sampling", value=False, visible=False)

        with gr.Group(visible=False) as emo_vec_group:
            with gr.Row():
                vec1 = gr.Slider(label="😊 Joy", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                vec2 = gr.Slider(label="😠 Anger", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                vec3 = gr.Slider(label="😢 Sadness", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                vec4 = gr.Slider(label="😨 Fear", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
            with gr.Row():
                vec5 = gr.Slider(label="🤢 Disgust", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                vec6 = gr.Slider(label="😔 Low Mood", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                vec7 = gr.Slider(label="😲 Surprise", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                vec8 = gr.Slider(label="😌 Calm", minimum=0.0, maximum=1.4, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            emo_text = gr.Textbox(
                label="Emotion Description",
                placeholder="e.g., happy, angry, excited, calm",
                value=""
            )

        def on_emo_mode_change(mode):
            return (
                gr.update(visible=(mode == 1)),  # emo_ref_group
                gr.update(visible=(mode in [1, 2])),  # emo_random
                gr.update(visible=(mode == 2)),  # emo_vec_group
                gr.update(visible=(mode == 3)),  # emo_text_group
            )

        emo_mode.change(
            on_emo_mode_change, [emo_mode],
            [emo_ref_group, emo_random, emo_vec_group, emo_text_group]
        )

    # Advanced Settings
    with gr.Accordion("⚙️ Advanced Settings", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Sampling Parameters**")
                do_sample = gr.Checkbox(label="Enable Sampling", value=True)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                top_p = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                top_k = gr.Slider(label="Top-K", minimum=0, maximum=100, value=30, step=1)
            with gr.Column():
                gr.Markdown("**Generation Parameters**")
                num_beams = gr.Slider(label="Num Beams", value=3, minimum=1, maximum=10, step=1)
                repetition_penalty = gr.Slider(label="Repetition Penalty", value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                length_penalty = gr.Slider(label="Length Penalty", value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                max_mel_tokens = gr.Slider(label="Max Mel Tokens", value=1500, minimum=50, maximum=3000, step=10)
            with gr.Column():
                gr.Markdown("**Text Splitting**")
                max_tokens_per_sentence = gr.Slider(
                    label="Max Tokens Per Sentence",
                    value=120, minimum=20, maximum=300, step=2,
                    info="Higher = longer sentences (80-200 recommended)"
                )
                sentences_preview = gr.Dataframe(
                    headers=["#", "Sentence", "Tokens"],
                    label="Sentence Preview",
                    wrap=True,
                )

    # Sentence preview update
    input_text.change(
        preview_sentences, [input_text, max_tokens_per_sentence], [sentences_preview]
    )
    max_tokens_per_sentence.change(
        preview_sentences, [input_text, max_tokens_per_sentence], [sentences_preview]
    )

    # Examples
    if example_cases:
        with gr.Accordion("📋 Examples", open=False):
            gr.Examples(
                examples=example_cases,
                inputs=[
                    prompt_audio, emo_mode, input_text,
                    emo_ref_audio, emo_weight, emo_text,
                    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                ],
                examples_per_page=10,
            )

    # System info
    with gr.Accordion("ℹ️ System Info", open=False):
        device_info = f"**Device:** {tts.device}"
        quant_info = f"**Quantization:** {'INT8' if tts.quantize else 'OFF'}"
        try:
            import psutil
            ram_info = f"**RAM:** {psutil.virtual_memory().total / 1024**3:.1f} GB total"
        except ImportError:
            ram_info = ""
        gpu_info = ""
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            gpu_info = f"**GPU:** {p.name} ({p.total_memory / 1024**3:.1f} GB)"
        gr.Markdown(f"{device_info} · {quant_info}\n\n{ram_info}\n\n{gpu_info}")

    # Wire up the generate button
    # Set default values for hidden components
    emo_ref_audio_default = gr.State(None)
    emo_weight_default = gr.State(0.8)
    emo_text_default = gr.State("")

    gen_button.click(
        generate_speech,
        inputs=[
            prompt_audio, input_text, emo_mode,
            emo_ref_audio, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random,
            max_tokens_per_sentence,
            do_sample, top_p, top_k, temperature,
            length_penalty, num_beams, repetition_penalty, max_mel_tokens,
        ],
        outputs=[output_audio],
    )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print_banner("WebUI Ready!")
    print(f"  Open in browser: http://localhost:{cmd_args.port}")
    print(f"  Device: {tts.device}")
    print(f"  Quantization: {'INT8' if tts.quantize else 'OFF'}")
    print()

    demo.queue(max_size=4).launch(
        server_name=cmd_args.host,
        server_port=cmd_args.port,
        share=cmd_args.share,
        show_error=True,
    )
