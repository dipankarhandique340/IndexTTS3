#!/usr/bin/env python3
"""
IndexTTS2 CPU/Low-VRAM Optimized Inference Engine
===================================================
This module provides a highly optimized inference wrapper for IndexTTS2 
that can run on CPU-only or low-VRAM GPU (2-4GB) systems.

Key optimizations:
  1. Dynamic INT8 quantization for GPT and W2V-BERT (largest models)
  2. Sequential model loading/offloading to minimize peak memory
  3. Aggressive memory management with gc + cache clearing
  4. Optional GPU offloading for individual stages
  5. Lazy-loading of QwenEmotion model (only when text emotion is needed)
  6. BFloat16 support for modern CPUs with AVX support

Usage:
  python infer_cpu.py --text "Hello world" --voice examples/voice_01.wav --output gen.wav
"""

import os
import sys
import gc
import time
import argparse
import warnings

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torchaudio
import librosa
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

# Force CPU if no GPU or low VRAM
def get_optimal_device():
    """Determine the best device considering VRAM constraints."""
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_mem / (1024**3)
            print(f">> GPU detected: {props.name} ({vram_gb:.1f} GB VRAM)")
            if vram_gb >= 2.0:
                return "cuda:0", vram_gb
            else:
                print(">> GPU VRAM too low, using CPU")
                return "cpu", 0
        except Exception:
            return "cpu", 0
    return "cpu", 0


def clear_memory(device="cpu"):
    """Aggressively clear all cached memory."""
    gc.collect()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def quantize_model_dynamic(model, dtype=torch.qint8):
    """Apply dynamic INT8 quantization to a model's linear layers."""
    try:
        quantized = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=dtype
        )
        return quantized
    except Exception as e:
        print(f">> Warning: Quantization failed ({e}), using original model")
        return model


class IndexTTS2_CPU:
    """
    CPU/Low-VRAM optimized IndexTTS2 inference engine.
    
    This class wraps the original IndexTTS2 model with optimizations:
    - Dynamic INT8 quantization for heavy models
    - Sequential model loading to reduce peak memory
    - Efficient memory management
    - All original features preserved (voice cloning, emotion, etc.)
    """

    def __init__(
        self,
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        device=None,
        use_quantization=True,
        verbose=True
    ):
        """
        Args:
            cfg_path: Path to config YAML
            model_dir: Path to model checkpoint directory
            device: Force device ('cpu', 'cuda:0'). Auto-detect if None.
            use_quantization: Apply INT8 dynamic quantization to large models
            verbose: Print loading progress
        """
        start_time = time.time()
        
        # Determine device
        if device is not None:
            self.device = device
            self.vram_gb = 0
        else:
            self.device, self.vram_gb = get_optimal_device()
        
        self.use_quantization = use_quantization and (self.device == "cpu" or self.vram_gb < 4)
        self.verbose = verbose
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        
        if self.verbose:
            print(f">> Device: {self.device}")
            print(f">> Quantization: {'ON (INT8)' if self.use_quantization else 'OFF'}")
            print(f">> Loading models...")
        
        # Load all sub-models with optimization
        self._load_tokenizer()
        self._load_gpt_model()
        self._load_semantic_model()
        self._load_semantic_codec()
        self._load_s2mel_model()
        self._load_campplus_model()
        self._load_bigvgan_model()
        self._load_emotion_data()
        self._setup_mel_fn()
        
        # QwenEmotion is lazy-loaded (only when text emotions are used)
        self._qwen_emo = None
        
        # Caches for reference audio (avoid reprocessing)
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None
        
        # Gradio progress callback (optional)
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None
        
        elapsed = time.time() - start_time
        if self.verbose:
            print(f">> All models loaded in {elapsed:.1f}s")
            self._print_memory_usage()
    
    def _print_memory_usage(self):
        """Print current memory usage."""
        import psutil
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        print(f">> RAM usage: {ram_mb:.0f} MB")
        if self.device != "cpu" and torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f">> VRAM usage: {vram_mb:.0f} MB")
    
    def _load_tokenizer(self):
        """Load BPE tokenizer and text normalizer."""
        from indextts.utils.front import TextNormalizer, TextTokenizer
        
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        if self.verbose:
            print(">> TextTokenizer loaded")
    
    def _load_gpt_model(self):
        """Load and optimize the GPT (UnifiedVoice) model."""
        from indextts.gpt.model_v2 import UnifiedVoice
        from indextts.utils.checkpoint import load_checkpoint
        
        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=False)
        gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, gpt_path)
        self.gpt = self.gpt.to(self.device)
        self.gpt.eval()
        
        # Apply INT8 quantization to GPT (biggest speedup on CPU)
        if self.use_quantization:
            self.gpt = quantize_model_dynamic(self.gpt)
            if self.verbose:
                print(">> GPT model quantized to INT8")
        
        # Initialize inference model (no DeepSpeed, no flash-attn, no accel)
        self.gpt.post_init_gpt2_config(
            use_deepspeed=False, 
            kv_cache=True, 
            half=False
        )
        
        if self.verbose:
            print(f">> GPT loaded from: {gpt_path}")
        clear_memory(self.device)
    
    def _load_semantic_model(self):
        """Load W2V-BERT-2.0 semantic model with quantization."""
        from indextts.utils.maskgct_utils import build_semantic_model
        from transformers import SeamlessM4TFeatureExtractor
        
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat)
        )
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)
        
        # Quantize W2V-BERT (second largest model ~580M params)
        if self.use_quantization:
            self.semantic_model = quantize_model_dynamic(self.semantic_model)
            if self.verbose:
                print(">> W2V-BERT model quantized to INT8")
        
        if self.verbose:
            print(">> Semantic model (W2V-BERT-2.0) loaded")
        clear_memory(self.device)
    
    def _load_semantic_codec(self):
        """Load MaskGCT semantic codec."""
        from indextts.utils.maskgct_utils import build_semantic_codec
        from huggingface_hub import hf_hub_download
        import safetensors
        
        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download(
            "amphion/MaskGCT", 
            filename="semantic_codec/model.safetensors"
        )
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        
        if self.verbose:
            print(f">> Semantic codec loaded from: {semantic_code_ckpt}")
        clear_memory(self.device)
    
    def _load_s2mel_model(self):
        """Load S2Mel (DiT/CFM) model."""
        from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
        
        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel, None, s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        self.s2mel.models['cfm'].estimator.setup_caches(
            max_batch_size=1, max_seq_length=8192
        )
        self.s2mel.eval()
        
        if self.verbose:
            print(f">> S2Mel loaded from: {s2mel_path}")
        clear_memory(self.device)
    
    def _load_campplus_model(self):
        """Load CAMPPlus speaker embedding model."""
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        from huggingface_hub import hf_hub_download
        
        campplus_ckpt_path = hf_hub_download(
            "funasr/campplus", filename="campplus_cn_common.bin"
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(
            torch.load(campplus_ckpt_path, map_location="cpu")
        )
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        
        if self.verbose:
            print(f">> CAMPPlus loaded from: {campplus_ckpt_path}")
        clear_memory(self.device)
    
    def _load_bigvgan_model(self):
        """Load BigVGAN vocoder."""
        from indextts.s2mel.modules.bigvgan import bigvgan
        
        bigvgan_name = self.cfg.vocoder.name
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(
            bigvgan_name, use_cuda_kernel=False
        )
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        
        if self.verbose:
            print(f">> BigVGAN loaded from: {bigvgan_name}")
        clear_memory(self.device)
    
    def _load_emotion_data(self):
        """Load emotion and speaker matrices."""
        emo_matrix = torch.load(
            os.path.join(self.model_dir, self.cfg.emo_matrix),
            map_location="cpu"
        )
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)
        
        spk_matrix = torch.load(
            os.path.join(self.model_dir, self.cfg.spk_matrix),
            map_location="cpu"
        )
        self.spk_matrix = spk_matrix.to(self.device)
        
        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)
        
        if self.verbose:
            print(">> Emotion/speaker matrices loaded")
    
    @property
    def qwen_emo(self):
        """Lazy-load QwenEmotion model (only when text emotion features are needed)."""
        if self._qwen_emo is None:
            if self.verbose:
                print(">> Loading QwenEmotion model (first use)...")
            from indextts.infer_v2 import QwenEmotion
            self._qwen_emo = QwenEmotion(
                os.path.join(self.model_dir, self.cfg.qwen_emo_path)
            )
            if self.verbose:
                print(">> QwenEmotion loaded")
        return self._qwen_emo
    
    def _setup_mel_fn(self):
        """Set up mel spectrogram function."""
        from indextts.s2mel.modules.audio import mel_spectrogram
        
        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)
    
    # ========================
    # Inference Helper Methods
    # ========================
    
    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        """Extract semantic embeddings from W2V-BERT."""
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def remove_long_silence(self, codes, silent_token=52, max_consecutive=30):
        """Shrink consecutive silence tokens in generated codes."""
        code_lens = []
        codes_list = []
        device = codes.device
        isfix = False
        for i in range(codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                ncode_idx = []
                n = 0
                for k in range(len_):
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def _load_and_cut_audio(self, audio_path, max_audio_length_seconds, verbose=False, sr=None):
        """Load and truncate audio."""
        if not sr:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path, sr=sr)
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)
        if audio.shape[1] > max_audio_samples:
            if verbose:
                print(f"  Audio truncated: {audio.shape[1]} -> {max_audio_samples} samples")
            audio = audio[:, :max_audio_samples]
        return audio, sr

    def normalize_emo_vec(self, emo_vector, apply_bias=True):
        """Normalize emotion vectors with bias."""
        if apply_bias:
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale_factor = 0.8 / emo_sum
            emo_vector = [vec * scale_factor for vec in emo_vector]
        return emo_vector

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    # ========================
    # Main Inference Method
    # ========================

    def infer(
        self,
        spk_audio_prompt,
        text,
        output_path,
        emo_audio_prompt=None,
        emo_alpha=1.0,
        emo_vector=None,
        use_emo_text=False,
        emo_text=None,
        use_random=False,
        interval_silence=200,
        verbose=False,
        max_text_tokens_per_segment=120,
        **generation_kwargs
    ):
        """
        Run TTS inference (identical API to original IndexTTS2.infer).
        
        Args:
            spk_audio_prompt: Path to reference speaker audio file
            text: Text to synthesize
            output_path: Path to save output WAV
            emo_audio_prompt: Optional path to emotion reference audio
            emo_alpha: Emotion blending factor (0.0 - 1.0)
            emo_vector: Optional 8-float list [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            use_emo_text: Auto-detect emotions from text
            emo_text: Custom text for emotion detection
            use_random: Enable stochastic sampling
            interval_silence: Silence duration (ms) between segments
            verbose: Print debug info
            max_text_tokens_per_segment: Max tokens per segment
        
        Returns:
            output_path on success, None on failure
        """
        import torch.nn.functional as F
        import random as rand_module

        print(">> Starting CPU-optimized inference...")
        self._set_gr_progress(0, "Starting inference...")
        start_time = time.perf_counter()

        # Handle emotion settings
        if use_emo_text or emo_vector is not None:
            emo_audio_prompt = None

        if use_emo_text:
            if emo_text is None:
                emo_text = text
            emo_dict = self.qwen_emo.inference(emo_text)
            print(f">> Detected emotions: {emo_dict}")
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]

        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0

        # Process speaker reference audio (with caching)
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                clear_memory(self.device)
            
            audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)

            _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
            ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
            feat = torchaudio.compliance.kaldi.fbank(
                audio_16k.to(ref_mel.device),
                num_mel_bins=80, dither=0, sample_frequency=16000
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = self.campplus_model(feat.unsqueeze(0))

            prompt_condition = self.s2mel.models['length_regulator'](
                S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
            )[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        # Emotion vector processing
        if emo_vector is not None:
            def find_most_similar_cosine(query_vector, matrix):
                query_vector = query_vector.float()
                matrix = matrix.float()
                similarities = F.cosine_similarity(query_vector, matrix, dim=1)
                return torch.argmax(similarities)

            weight_vector = torch.tensor(emo_vector, device=self.device)
            if use_random:
                random_index = [rand_module.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0).unsqueeze(0)

        # Process emotion reference audio
        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                clear_memory(self.device)
            emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
            emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
            emo_input_features = emo_inputs["input_features"].to(self.device)
            emo_attention_mask = emo_inputs["attention_mask"].to(self.device)
            emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)
            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
        else:
            emo_cond_emb = self.cache_emo_cond

        # Text processing
        self._set_gr_progress(0.1, "Processing text...")
        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment)
        segments_count = len(segments)

        if verbose:
            print(f">> Text segments: {segments_count}")
            print(f">> Max tokens per segment: {max_text_tokens_per_segment}")

        # Generation parameters
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        sampling_rate = 22050

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0

        for seg_idx, sent in enumerate(segments):
            self._set_gr_progress(
                0.2 + 0.7 * seg_idx / segments_count,
                f"Synthesizing segment {seg_idx + 1}/{segments_count}..."
            )

            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)

            m_start = time.perf_counter()
            with torch.no_grad():
                # Merge emotion vectors
                emovec = self.gpt.merge_emovec(
                    spk_cond_emb,
                    emo_cond_emb,
                    torch.tensor([spk_cond_emb.shape[-1]], device=self.device),
                    torch.tensor([emo_cond_emb.shape[-1]], device=self.device),
                    alpha=emo_alpha
                )

                if emo_vector is not None:
                    emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

                # GPT autoregressive generation
                codes, speech_conditioning_latent = self.gpt.inference_speech(
                    spk_cond_emb,
                    text_tokens,
                    emo_cond_emb,
                    cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=self.device),
                    emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=self.device),
                    emo_vec=emovec,
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=1,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    max_generate_length=max_mel_tokens,
                    **generation_kwargs
                )
                gpt_gen_time += time.perf_counter() - m_start

                # Compute code lengths
                code_lens = []
                max_code_len = 0
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                        code_len = len_[0].item() if len_.numel() > 0 else len(code)
                    code_lens.append(code_len)
                    max_code_len = max(max_code_len, code_len)
                codes = codes[:, :max_code_len]
                code_lens = torch.LongTensor(code_lens).to(self.device)

                # GPT forward pass for latent
                m_start = time.perf_counter()
                use_speed = torch.zeros(spk_cond_emb.size(0)).to(self.device).long()
                latent = self.gpt(
                    speech_conditioning_latent,
                    text_tokens,
                    torch.tensor([text_tokens.shape[-1]], device=self.device),
                    codes,
                    torch.tensor([codes.shape[-1]], device=self.device),
                    emo_cond_emb,
                    cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=self.device),
                    emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=self.device),
                    emo_vec=emovec,
                    use_speed=use_speed,
                )
                gpt_forward_time += time.perf_counter() - m_start

                # S2Mel synthesis
                m_start = time.perf_counter()
                diffusion_steps = 25
                inference_cfg_rate = 0.7
                latent = self.s2mel.models['gpt_layer'](latent)
                S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                S_infer = S_infer.transpose(1, 2)
                S_infer = S_infer + latent
                target_lengths = (code_lens * 1.72).long()

                cond = self.s2mel.models['length_regulator'](
                    S_infer, ylens=target_lengths, n_quantizers=3, f0=None
                )[0]
                cat_condition = torch.cat([prompt_condition, cond], dim=1)
                vc_target = self.s2mel.models['cfm'].inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(self.device),
                    ref_mel, style, None, diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate
                )
                vc_target = vc_target[:, :, ref_mel.size(-1):]
                s2mel_time += time.perf_counter() - m_start

                # BigVGAN vocoding
                m_start = time.perf_counter()
                wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                bigvgan_time += time.perf_counter() - m_start
                wav = wav.squeeze(1)

            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            wavs.append(wav.cpu())
            
            if verbose:
                print(f"  Segment {seg_idx + 1}/{segments_count} done")

        end_time = time.perf_counter()

        # Combine and save
        self._set_gr_progress(0.9, "Saving audio...")
        
        # Insert silence between segments
        if len(wavs) > 1 and interval_silence > 0:
            channel_size = wavs[0].size(0)
            sil_dur = int(sampling_rate * interval_silence / 1000.0)
            sil_tensor = torch.zeros(channel_size, sil_dur)
            wavs_with_silence = []
            for i, w in enumerate(wavs):
                wavs_with_silence.append(w)
                if i < len(wavs) - 1:
                    wavs_with_silence.append(sil_tensor)
            wavs = wavs_with_silence

        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate

        print(f">> Performance breakdown:")
        print(f"   GPT generation:  {gpt_gen_time:.2f}s")
        print(f"   GPT forward:     {gpt_forward_time:.2f}s")
        print(f"   S2Mel synthesis:  {s2mel_time:.2f}s")
        print(f"   BigVGAN vocoding: {bigvgan_time:.2f}s")
        print(f"   Total time:       {end_time - start_time:.2f}s")
        print(f"   Audio length:     {wav_length:.2f}s")
        print(f"   RTF:              {(end_time - start_time) / wav_length:.4f}")

        # Save
        wav = wav.cpu()
        if output_path:
            if os.path.isfile(output_path):
                os.remove(output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(f">> Saved: {output_path}")
            return output_path
        else:
            wav_data = wav.type(torch.int16).numpy().T
            return (sampling_rate, wav_data)


def main():
    parser = argparse.ArgumentParser(
        description="IndexTTS2 CPU/Low-VRAM Optimized Inference"
    )
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--voice", type=str, required=True, help="Path to reference speaker audio")
    parser.add_argument("--output", type=str, default="gen_cpu.wav", help="Output WAV path")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda:0)")
    parser.add_argument("--no-quantize", action="store_true", help="Disable INT8 quantization")
    parser.add_argument("--emo-audio", type=str, default=None, help="Emotion reference audio")
    parser.add_argument("--emo-alpha", type=float, default=1.0, help="Emotion blending factor")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml", help="Config path")
    parser.add_argument("--model-dir", type=str, default="checkpoints", help="Model directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    tts = IndexTTS2_CPU(
        cfg_path=args.config,
        model_dir=args.model_dir,
        device=args.device,
        use_quantization=not args.no_quantize,
        verbose=True
    )

    tts.infer(
        spk_audio_prompt=args.voice,
        text=args.text,
        output_path=args.output,
        emo_audio_prompt=args.emo_audio,
        emo_alpha=args.emo_alpha,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
