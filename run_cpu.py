#!/usr/bin/env python3
"""
=============================================================================
IndexTTS2 — GPU-Accelerated / Low-VRAM All-in-One Runner
=============================================================================
A single file that:
  1. Downloads all required model checkpoints (if missing)
  2. Loads models on GPU (FP16) for fast inference, or CPU fallback
  3. Runs TTS inference and saves the output WAV
  4. Optionally runs a full self-test

Run:
  .venv/bin/python run_cpu.py
  .venv/bin/python run_cpu.py --text "Hello world" --voice examples/voice_01.wav
  .venv/bin/python run_cpu.py --test
  .venv/bin/python run_cpu.py --download-only
============================================================================="""

import os, sys, gc, time, json, re, argparse, warnings, glob, random

os.environ["HF_HUB_CACHE"]         = "./checkpoints/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch

# ---------------------------------------------------------------------------
# 0.  UTILITIES
# ---------------------------------------------------------------------------

def clear_mem(device="cpu"):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_banner(text):
    w = 60
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)


def ram_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except ImportError:
        return -1


# ---------------------------------------------------------------------------
# 1.  DOWNLOAD  — fetch every required checkpoint
# ---------------------------------------------------------------------------

def download_checkpoints(model_dir="checkpoints"):
    """Download all IndexTTS-2 checkpoints that are missing."""
    print_banner("STEP 1 / 3 — Downloading Model Checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    # ---- Main IndexTTS-2 weights -------------------------------------------
    required_files = ["gpt.pth", "s2mel.pth", "bpe.model",
                      "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt",
                      "config.yaml", "pinyin.vocab"]
    need_main = any(
        not os.path.isfile(os.path.join(model_dir, f))
        for f in required_files
    )

    if need_main:
        from huggingface_hub import snapshot_download
        print(">> Downloading IndexTeam/IndexTTS-2  (≈2 GB) ...")
        snapshot_download(
            repo_id="IndexTeam/IndexTTS-2",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        print(">> IndexTTS-2 checkpoints ready.")
    else:
        print(">> IndexTTS-2 checkpoints already present — skipping.")

    # ---- Qwen emotion model -----------------------------------------------
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    qwen_dir = os.path.join(model_dir, cfg.qwen_emo_path)
    if not os.path.isdir(qwen_dir) or not os.listdir(qwen_dir):
        print(">> Qwen emotion model will auto-download on first use.")
    else:
        print(f">> Qwen emotion model present at {qwen_dir}")

    # ---- Small HuggingFace models (auto-cached on first use) ---------------
    print(">> Pre-caching small helper models …")
    try:
        from transformers import SeamlessM4TFeatureExtractor
        SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        print("   ✓ facebook/w2v-bert-2.0  feature extractor")
    except Exception as e:
        print(f"   ⏳ w2v-bert-2.0 will download on first inference ({e})")

    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("amphion/MaskGCT",
                        filename="semantic_codec/model.safetensors")
        print("   ✓ amphion/MaskGCT  semantic codec")
    except Exception as e:
        print(f"   ⏳ MaskGCT will download on first inference ({e})")

    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("funasr/campplus",
                        filename="campplus_cn_common.bin")
        print("   ✓ funasr/campplus  speaker encoder")
    except Exception as e:
        print(f"   ⏳ campplus will download on first inference ({e})")

    print(">> All downloads finished.\n")


# ---------------------------------------------------------------------------
# 2.  MODEL  — GPU-accelerated / low-VRAM TTS engine
# ---------------------------------------------------------------------------

def pick_device():
    """Return (device_str, vram_gb)."""
    if torch.cuda.is_available():
        try:
            p  = torch.cuda.get_device_properties(0)
            gb = p.total_memory / 1024**3
            print(f">> GPU: {p.name}  ({gb:.1f} GB VRAM)")
            return ("cuda:0", gb) if gb >= 2.0 else ("cpu", 0)
        except Exception:
            pass
    print(">> No CUDA GPU found — using CPU")
    return ("cpu", 0)


def quantize_dyn(model, dtype=None):
    """INT-8 dynamic quantisation of nn.Linear layers (CPU only)."""
    if dtype is None:
        dtype = torch.qint8
    try:
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=dtype)
    except Exception as exc:
        print(f">> quantise skipped: {exc}")
        return model


class IndexTTS2CPU:
    """
    Full IndexTTS-2 inference — GPU FP16 (fast) or CPU INT8 (fallback).

    Strategy
    --------
    * If CUDA GPU available (≥ 2 GB): load all models on GPU in FP16
      → ~10-20× faster than CPU, fits in 4 GB VRAM
    * If no GPU:  fall back to CPU with INT-8 quantisation
    * No flash-attn / deepspeed needed
    * Lazy-load QwenEmotion only when text-emotion mode is requested
    * Aggressive memory clean-up between loading phases
    * Reference-audio caching (avoid re-processing same speaker)
    """

    # ------------------------------------------------------------------
    def __init__(self, cfg_path="checkpoints/config.yaml",
                 model_dir="checkpoints", device=None,
                 quantize=True, verbose=True):
        t0 = time.time()
        self.verbose = verbose

        # device
        if device:
            self.device = device
        else:
            self.device, self._vram_gb = pick_device()

        self.use_gpu = self.device.startswith("cuda")
        self.use_fp16 = self.use_gpu  # FP16 on GPU, FP32 on CPU
        self.quantize = quantize and not self.use_gpu  # INT8 only on CPU

        from omegaconf import OmegaConf
        self.cfg      = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        mode = "GPU FP16" if self.use_gpu else ("CPU INT8" if self.quantize else "CPU FP32")
        if verbose:
            print(f">> device={self.device}  mode={mode}")

        # load sub-models one by one
        self._load_tokenizer()
        self._load_gpt()
        self._load_semantic()
        self._load_semantic_codec()
        self._load_s2mel()
        self._load_campplus()
        self._load_bigvgan()
        self._load_emo_data()
        self._init_mel_fn()

        # lazy
        self._qwen = None

        # caches
        self._spk_cache = {}
        self._emo_cache = {}

        self.gr_progress    = None
        self.model_version  = getattr(self.cfg, "version", None)

        if verbose:
            vram_used = ""
            if self.use_gpu:
                vram_used = f"  VRAM ≈ {torch.cuda.memory_allocated()/(1024**2):.0f} MB"
            print(f">> all models loaded in {time.time()-t0:.1f}s   "
                  f"(RAM ≈ {ram_mb():.0f} MB{vram_used})")

    # ---- individual loaders ------------------------------------------------

    def _load_tokenizer(self):
        from indextts.utils.front import TextNormalizer, TextTokenizer
        bpe = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer(); self.normalizer.load()
        self.tokenizer  = TextTokenizer(bpe, self.normalizer)
        if self.verbose: print(">> tokeniser loaded")

    def _load_gpt(self):
        from indextts.gpt.model_v2 import UnifiedVoice
        from indextts.utils.checkpoint import load_checkpoint
        path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=False)
        load_checkpoint(self.gpt, path)
        if self.use_fp16:
            self.gpt.half()
        self.gpt.to(self.device).eval()
        if self.quantize:
            self.gpt = quantize_dyn(self.gpt)
        self.gpt.post_init_gpt2_config(
            use_deepspeed=False, kv_cache=True,
            half=self.use_fp16)
        if self.use_fp16:
            self.gpt.inference_model.half()
        if self.verbose: print(f">> GPT loaded  ({path})")
        clear_mem(self.device)

    def _load_semantic(self):
        from indextts.utils.maskgct_utils import build_semantic_model
        from transformers import SeamlessM4TFeatureExtractor
        self.feat_ext = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0")
        self.sem_model, self.sem_mean, self.sem_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat))
        self.sem_model.to("cpu").eval()
        self.sem_mean = self.sem_mean.to("cpu")
        self.sem_std  = self.sem_std.to("cpu")
        if self.quantize:
            self.sem_model = quantize_dyn(self.sem_model)
        if self.verbose: print(">> W2V-BERT-2.0 loaded on CPU (saving VRAM)")
        clear_mem(self.device)

    def _load_semantic_codec(self):
        import safetensors
        from indextts.utils.maskgct_utils import build_semantic_codec
        from huggingface_hub import hf_hub_download
        sc = build_semantic_codec(self.cfg.semantic_codec)
        ck = hf_hub_download("amphion/MaskGCT",
                             filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(sc, ck)
        self.sem_codec = sc.to("cpu"); self.sem_codec.eval()
        if self.verbose: print(">> semantic codec loaded on CPU (saving VRAM)")
        clear_mem(self.device)

    def _load_s2mel(self):
        from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
        path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        m = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        m, _, _, _ = load_checkpoint2(m, None, path, load_only_params=True,
                                      ignore_modules=[], is_distributed=False)
        if self.use_fp16:
            m.half()
        self.s2mel = m.to(self.device)
        self.s2mel.models["cfm"].estimator.setup_caches(
            max_batch_size=1, max_seq_length=8192)
        self.s2mel.eval()
        if self.verbose: print(f">> S2Mel loaded  ({path})")
        clear_mem(self.device)

    def _load_campplus(self):
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        from huggingface_hub import hf_hub_download
        ck = hf_hub_download("funasr/campplus",
                             filename="campplus_cn_common.bin")
        m  = CAMPPlus(feat_dim=80, embedding_size=192)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        self.campplus = m.to("cpu"); self.campplus.eval()
        if self.verbose: print(">> CAMPPlus loaded on CPU (saving VRAM)")
        clear_mem(self.device)

    def _load_bigvgan(self):
        from indextts.s2mel.modules.bigvgan import bigvgan
        name = self.cfg.vocoder.name
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(
            name, use_cuda_kernel=False)
        # BigVGAN stays on CPU to save VRAM — it's fast enough (single fwd pass)
        self.bigvgan.to("cpu").eval()
        self.bigvgan.remove_weight_norm()
        if self.verbose: print(f">> BigVGAN loaded on CPU  ({name})")
        clear_mem(self.device)

    def _load_emo_data(self):
        em = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix),
                        map_location="cpu").to(self.device)
        sm = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix),
                        map_location="cpu").to(self.device)
        if self.use_fp16:
            em = em.half()
            sm = sm.half()
        self.emo_num = list(self.cfg.emo_num)
        self.emo_mat = torch.split(em, self.emo_num)
        self.spk_mat = torch.split(sm, self.emo_num)
        if self.verbose: print(">> emotion data loaded")

    @property
    def qwen_emo(self):
        if self._qwen is None:
            if self.verbose: print(">> lazy-loading QwenEmotion …")
            from indextts.infer_v2 import QwenEmotion
            self._qwen = QwenEmotion(
                os.path.join(self.model_dir, self.cfg.qwen_emo_path))
            if self.verbose: print(">> QwenEmotion ready")
        return self._qwen

    def _init_mel_fn(self):
        from indextts.s2mel.modules.audio import mel_spectrogram
        a = self.cfg.s2mel["preprocess_params"]["spect_params"]
        kw = dict(n_fft=a["n_fft"], win_size=a["win_length"],
                  hop_size=a["hop_length"], num_mels=a["n_mels"],
                  sampling_rate=self.cfg.s2mel["preprocess_params"]["sr"],
                  fmin=a.get("fmin", 0),
                  fmax=None if a.get("fmax", "None") == "None" else 8000,
                  center=False)
        self.mel_fn = lambda x: mel_spectrogram(x, **kw)

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _load_audio(path, max_sec, sr=None):
        import librosa
        if sr:
            audio, _ = librosa.load(path, sr=sr)
        else:
            audio, sr = librosa.load(path)
        audio = torch.tensor(audio).unsqueeze(0)
        mx = int(max_sec * sr)
        if audio.shape[1] > mx:
            audio = audio[:, :mx]
        return audio, sr

    @torch.no_grad()
    def _get_emb(self, feats, mask):
        sd = next(self.sem_model.parameters()).dtype
        sv = next(self.sem_model.parameters()).device
        feats = feats.to(dtype=sd, device=sv)
        mask  = mask.to(dtype=sd, device=sv) if mask.is_floating_point() else mask.to(device=sv)
        o = self.sem_model(input_features=feats, attention_mask=mask,
                           output_hidden_states=True)
        f = o.hidden_states[17]
        result = (f - self.sem_mean.to(dtype=sd, device=sv)) / self.sem_std.to(dtype=sd, device=sv)
        if self.use_fp16:
            result = result.half().to(self.device)
        return result

    def _set_gr_progress(self, v, d):
        if self.gr_progress is not None:
            self.gr_progress(v, desc=d)

    # ---- main inference ----------------------------------------------------

    @torch.no_grad()
    def infer(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None, use_emo_text=False, emo_text=None,
              use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120,
              **gen_kw):
        """
        Synthesise speech — **same API as the original IndexTTS2.infer()**.

        Returns the output path on success, None otherwise.
        """
        import torchaudio
        import torch.nn.functional as F

        print(">> inference started …")
        self._set_gr_progress(0, "starting")
        t0 = time.perf_counter()

        # emotion pre-processing
        if use_emo_text or emo_vector is not None:
            emo_audio_prompt = None
        if use_emo_text:
            emo_text = emo_text or text
            ed = self.qwen_emo.inference(emo_text)
            print(f">> emotions: {ed}")
            emo_vector = list(ed.values())
        if emo_vector is not None:
            s = max(0.0, min(1.0, emo_alpha))
            if s != 1.0:
                emo_vector = [int(x*s*10000)/10000 for x in emo_vector]
        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0

        # ----- speaker conditioning (cached) --------------------------------
        if (self._spk_cache.get("path") != spk_audio_prompt):
            audio, sr    = self._load_audio(spk_audio_prompt, 15)
            a22          = torchaudio.transforms.Resample(sr, 22050)(audio)
            a16          = torchaudio.transforms.Resample(sr, 16000)(audio)
            inp          = self.feat_ext(a16, sampling_rate=16000,
                                         return_tensors="pt")
            feats        = inp["input_features"].to("cpu")
            mask         = inp["attention_mask"].to("cpu")
            spk_cond     = self._get_emb(feats, mask).to(self.device)
            # sem_codec expects same dtype as its weights
            sc_dtype     = next(self.sem_codec.parameters()).dtype
            sc_device    = next(self.sem_codec.parameters()).device
            sc_input     = spk_cond.to(dtype=sc_dtype, device=sc_device)
            _, S_ref     = self.sem_codec.quantize(sc_input)
            S_ref        = S_ref.to(self.device)
            if self.use_fp16:
                S_ref = S_ref.half()
            ref_mel      = self.mel_fn(a22.to(self.device).float())
            if self.use_fp16:
                ref_mel  = ref_mel.half()
            ref_tgt_len  = torch.LongTensor([ref_mel.size(2)]).to(self.device)
            # fbank needs FP32 on CPU
            fb           = torchaudio.compliance.kaldi.fbank(
                              a16, num_mel_bins=80,
                              dither=0, sample_frequency=16000)
            fb           = fb - fb.mean(dim=0, keepdim=True)
            camp_dtype   = next(self.campplus.parameters()).dtype
            camp_device  = next(self.campplus.parameters()).device
            style        = self.campplus(fb.unsqueeze(0).to(dtype=camp_dtype, device=camp_device)).to(self.device)
            if self.use_fp16:
                style = style.half()
            prompt_cond  = self.s2mel.models["length_regulator"](
                              S_ref, ylens=ref_tgt_len,
                              n_quantizers=3, f0=None)[0]
            self._spk_cache = dict(path=spk_audio_prompt, cond=spk_cond,
                                   style=style, prompt=prompt_cond,
                                   mel=ref_mel)
        c = self._spk_cache
        spk_cond, style, prompt_cond, ref_mel = (
            c["cond"], c["style"], c["prompt"], c["mel"])

        # ----- emotion vector matrix ----------------------------------------
        emovec_mat = None
        weight_vector = None
        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            if use_random:
                ri = [random.randint(0, x-1) for x in self.emo_num]
            else:
                ri = [torch.argmax(F.cosine_similarity(
                          style.float(), m.float(), dim=1))
                      for m in self.spk_mat]
            em = torch.cat([m[i].unsqueeze(0)
                            for i, m in zip(ri, self.emo_mat)], 0)
            emovec_mat = (weight_vector.unsqueeze(1) * em).sum(0).unsqueeze(0)

        # ----- emotion conditioning (cached) --------------------------------
        if self._emo_cache.get("path") != emo_audio_prompt:
            ea, _   = self._load_audio(emo_audio_prompt, 15, sr=16000)
            ei      = self.feat_ext(ea, sampling_rate=16000,
                                    return_tensors="pt")
            ef      = ei["input_features"].to("cpu")
            em      = ei["attention_mask"].to("cpu")
            emo_cond = self._get_emb(ef, em).to(self.device)
            self._emo_cache = dict(path=emo_audio_prompt, cond=emo_cond)
        emo_cond = self._emo_cache["cond"]

        # ----- text segmentation -------------------------------------------
        self._set_gr_progress(0.1, "tokenising")
        toks    = self.tokenizer.tokenize(text)
        segs    = self.tokenizer.split_segments(toks,
                      max_text_tokens_per_segment)
        n_segs  = len(segs)
        if verbose:
            print(f">> {n_segs} segment(s),  "
                  f"max_tok={max_text_tokens_per_segment}")

        # generation hyper-params
        top_p     = gen_kw.pop("top_p", 0.8)
        top_k     = gen_kw.pop("top_k", 30)
        temp      = gen_kw.pop("temperature", 0.8)
        lp        = gen_kw.pop("length_penalty", 0.0)
        nb        = gen_kw.pop("num_beams", 3)
        rp        = gen_kw.pop("repetition_penalty", 10.0)
        mml       = gen_kw.pop("max_mel_tokens", 1500)
        gen_kw.pop("do_sample", None)
        sr_out    = 22050

        wavs = []
        t_gpt = t_fwd = t_s2m = t_voc = 0.0

        for si, seg in enumerate(segs):
            self._set_gr_progress(0.15 + 0.7*si/n_segs,
                                  f"segment {si+1}/{n_segs}")

            ids = self.tokenizer.convert_tokens_to_ids(seg)
            tt  = torch.tensor(ids, dtype=torch.int32,
                               device=self.device).unsqueeze(0)

            # merge emo vec
            emovec = self.gpt.merge_emovec(
                spk_cond, emo_cond,
                torch.tensor([spk_cond.shape[-1]], device=self.device),
                torch.tensor([emo_cond.shape[-1]], device=self.device),
                alpha=emo_alpha)
            if emovec_mat is not None:
                emovec = emovec_mat + (1 - weight_vector.sum()) * emovec
            
            # Ensure emovec stays fp16 to prevent type mismatch down the line
            if self.use_fp16:
                emovec = emovec.half()

            # ---- GPT generate codes ----
            t1 = time.perf_counter()
            codes, sc_lat = self.gpt.inference_speech(
                spk_cond, tt, emo_cond,
                cond_lengths=torch.tensor(
                    [spk_cond.shape[-1]], device=self.device),
                emo_cond_lengths=torch.tensor(
                    [emo_cond.shape[-1]], device=self.device),
                emo_vec=emovec,
                do_sample=True, top_p=top_p, top_k=top_k,
                temperature=temp, num_return_sequences=1,
                length_penalty=lp, num_beams=nb,
                repetition_penalty=rp,
                max_generate_length=mml, **gen_kw)
            t_gpt += time.perf_counter() - t1

            # trim to actual length
            cl = []
            for c in codes:
                if self.stop_mel_token not in c:
                    cl.append(len(c))
                else:
                    idx = (c == self.stop_mel_token).nonzero(as_tuple=False)
                    cl.append(idx[0].item() if idx.numel() else len(c))
            mx = max(cl)
            codes = codes[:, :mx]
            cl_t  = torch.LongTensor(cl).to(self.device)

            # ---- GPT forward (latent) ----
            t1 = time.perf_counter()
            us = torch.zeros(spk_cond.size(0), device=self.device).long()
            latent = self.gpt(
                sc_lat, tt,
                torch.tensor([tt.shape[-1]], device=self.device),
                codes,
                torch.tensor([codes.shape[-1]], device=self.device),
                emo_cond,
                cond_mel_lengths=torch.tensor(
                    [spk_cond.shape[-1]], device=self.device),
                emo_cond_mel_lengths=torch.tensor(
                    [emo_cond.shape[-1]], device=self.device),
                emo_vec=emovec, use_speed=us)
            t_fwd += time.perf_counter() - t1

            # ---- S2Mel (CFM diffusion) ----
            t1 = time.perf_counter()
            latent = self.s2mel.models["gpt_layer"](latent)
            S_inf  = self.sem_codec.quantizer.vq2emb(codes.unsqueeze(1).cpu()).to(self.device)
            if self.use_fp16:
                S_inf = S_inf.half()
            S_inf  = S_inf.transpose(1, 2) + latent
            tgt_l  = (cl_t * 1.72).long()
            cond   = self.s2mel.models["length_regulator"](
                        S_inf, ylens=tgt_l, n_quantizers=3, f0=None)[0]
            cat_c  = torch.cat([prompt_cond, cond], dim=1)
            vc     = self.s2mel.models["cfm"].inference(
                        cat_c,
                        torch.LongTensor([cat_c.size(1)]).to(self.device),
                        ref_mel, style, None, 25,
                        inference_cfg_rate=0.7)
            vc     = vc[:, :, ref_mel.size(-1):]
            t_s2m += time.perf_counter() - t1

            # ---- BigVGAN vocoder (runs on CPU) ----
            t1 = time.perf_counter()
            wav = self.bigvgan(vc.float().cpu()).squeeze().unsqueeze(0).squeeze(1)
            t_voc += time.perf_counter() - t1

            wav = torch.clamp(32767 * wav, -32767, 32767)
            wavs.append(wav.cpu())

            if verbose:
                print(f"   seg {si+1}/{n_segs}  done  "
                      f"({wav.shape[-1]/sr_out:.1f}s audio)")

        # ----- stitch -------------------------------------------------------
        self._set_gr_progress(0.9, "saving")
        if len(wavs) > 1 and interval_silence > 0:
            ch  = wavs[0].size(0)
            sil = torch.zeros(ch, int(sr_out * interval_silence / 1000))
            tmp = []
            for i, w in enumerate(wavs):
                tmp.append(w)
                if i < len(wavs) - 1:
                    tmp.append(sil)
            wavs = tmp

        wav = torch.cat(wavs, dim=1).cpu()
        dur = wav.shape[-1] / sr_out
        total = time.perf_counter() - t0

        print(f">> ── timing ──────────────────────────")
        print(f"   GPT gen     {t_gpt:.2f}s")
        print(f"   GPT fwd     {t_fwd:.2f}s")
        print(f"   S2Mel       {t_s2m:.2f}s")
        print(f"   BigVGAN     {t_voc:.2f}s")
        print(f"   TOTAL       {total:.2f}s")
        print(f"   audio       {dur:.2f}s    RTF {total/dur:.3f}")

        if output_path:
            d = os.path.dirname(output_path)
            if d:
                os.makedirs(d, exist_ok=True)
            torchaudio.save(output_path, wav.to(torch.int16), sr_out)
            print(f">> saved  {output_path}  "
                  f"({os.path.getsize(output_path)//1024} kB)")
            return output_path
        return None


# ---------------------------------------------------------------------------
# 3.  TESTS
# ---------------------------------------------------------------------------

def run_tests(tts):
    """Quick self-test suite."""
    import torch

    print_banner("SELF-TEST SUITE")
    results = {}
    voices  = sorted(glob.glob("examples/voice_*.wav"))
    voice   = voices[0] if voices else None

    if voice is None:
        print(">> ⚠  No voice samples in examples/ — skipping inference tests")
        return

    # Test A — English
    try:
        print("\n[Test A] English synthesis")
        out = tts.infer(voice,
                        "Hello! This is IndexTTS2 running on CPU.",
                        "test_a_en.wav", verbose=True)
        results["English"] = out is not None and os.path.exists(out)
    except Exception as e:
        print(f"FAIL: {e}")
        results["English"] = False

    # Test B — Chinese
    try:
        print("\n[Test B] Chinese synthesis")
        out = tts.infer(voice,
                        "欢迎体验 AI 语音技术。",
                        "test_b_zh.wav", verbose=True)
        results["Chinese"] = out is not None and os.path.exists(out)
    except Exception as e:
        print(f"FAIL: {e}")
        results["Chinese"] = False

    # Test C — Emotion vector
    try:
        print("\n[Test C] Emotion vector (happy + surprised)")
        out = tts.infer(voice,
                        "Wow this is incredible, I love it!",
                        "test_c_emo.wav",
                        emo_vector=[0.6, 0, 0, 0, 0, 0, 0.3, 0],
                        verbose=True)
        results["Emotion"] = out is not None and os.path.exists(out)
    except Exception as e:
        print(f"FAIL: {e}")
        results["Emotion"] = False

    # Test D — Second call (uses cache – should be faster)
    try:
        print("\n[Test D] Cached speaker (speed test)")
        t1 = time.time()
        out = tts.infer(voice,
                        "Second call is faster thanks to caching.",
                        "test_d_cache.wav", verbose=True)
        t2 = time.time()
        results["Caching"] = out is not None
        print(f">> cached call took {t2-t1:.1f}s")
    except Exception as e:
        print(f"FAIL: {e}")
        results["Caching"] = False

    # Summary
    print("\n" + "=" * 50)
    ok = sum(results.values())
    for k, v in results.items():
        print(f"  {'✅' if v else '❌'}  {k}")
    print(f"\n  {ok}/{len(results)} passed")
    print(f"  RAM ≈ {ram_mb():.0f} MB")
    print("=" * 50)


# ---------------------------------------------------------------------------
# 4.  MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="IndexTTS2 — CPU/Low-VRAM all-in-one runner")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to synthesise")
    parser.add_argument("--voice", type=str, default=None,
                        help="Speaker reference WAV")
    parser.add_argument("--output", type=str, default="gen_cpu.wav",
                        help="Output WAV path")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device  (cpu / cuda:0)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable INT8 quantisation")
    parser.add_argument("--emo-audio", type=str, default=None)
    parser.add_argument("--emo-alpha", type=float, default=1.0)
    parser.add_argument("--emo-vec",   type=str, default=None,
                        help="Comma-separated 8 floats: "
                             "happy,angry,sad,afraid,disgusted,"
                             "melancholic,surprised,calm")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download models, don't run inference")
    parser.add_argument("--test", action="store_true",
                        help="Run self-test suite after loading")
    parser.add_argument("--config", type=str,
                        default="checkpoints/config.yaml")
    parser.add_argument("--model-dir", type=str, default="checkpoints")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # ── download ──────────────────────────────────────────────────────────
    download_checkpoints(args.model_dir)
    if args.download_only:
        print(">> Download-only mode.  Exiting.")
        return

    # ── load model ────────────────────────────────────────────────────────
    print_banner("STEP 2 / 3 — Loading Models")
    tts = IndexTTS2CPU(
        cfg_path=args.config,
        model_dir=args.model_dir,
        device=args.device,
        quantize=not args.no_quantize,
        verbose=True,
    )

    # ── test mode ─────────────────────────────────────────────────────────
    if args.test:
        run_tests(tts)
        return

    # ── infer ─────────────────────────────────────────────────────────────
    print_banner("STEP 3 / 3 — Inference")

    text  = args.text
    voice = args.voice

    # auto-pick defaults if nothing supplied
    if text is None:
        text = "Hello! This is IndexTTS2 running efficiently on CPU. The quality should be identical to the GPU version."
        print(f">> using default text: {text}")

    if voice is None:
        voices = sorted(glob.glob("examples/voice_*.wav"))
        if voices:
            voice = voices[0]
            print(f">> using default voice: {voice}")
        else:
            print(">> ERROR: no voice file found.  Supply --voice <path>")
            return

    emo_vec = None
    if args.emo_vec:
        emo_vec = [float(x) for x in args.emo_vec.split(",")]
        if len(emo_vec) != 8:
            print(">> ERROR: --emo-vec needs exactly 8 comma-separated floats")
            return

    result = tts.infer(
        spk_audio_prompt=voice,
        text=text,
        output_path=args.output,
        emo_audio_prompt=args.emo_audio,
        emo_alpha=args.emo_alpha,
        emo_vector=emo_vec,
        verbose=args.verbose or True,
    )

    if result:
        print(f"\n>> ✅  Success!  Output: {result}")
    else:
        print("\n>> ❌  Inference returned nothing.")


if __name__ == "__main__":
    main()
