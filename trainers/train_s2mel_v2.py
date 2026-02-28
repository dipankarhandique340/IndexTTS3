#!/usr/bin/env python3
"""
End-to-end finetuning for IndexTTS2 S2Mel (DiT flow-matching) module.

Architecture: semantic_codes → length_regulator → 512-dim cond → CFM(DiT) → mel
The full pipeline uses MyModel which wraps both length_regulator and CFM.

Training data:
- Audio files → mel spectrograms (targets)
- Pre-extracted semantic codes (.npy from preprocessing)
- Style (extracted from audio via CAMPPlus)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from indextts.s2mel.modules.audio import mel_spectrogram

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train IndexTTS2 S2Mel (DiT).")
    p.add_argument("--train-manifest", dest="train_manifests", action="append", type=str, required=True)
    p.add_argument("--val-manifest", dest="val_manifests", action="append", type=str, required=True)
    p.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"))
    p.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/s2mel.pth"))
    p.add_argument("--output-dir", type=Path, default=Path("trained_ckpts_s2mel"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accumulation", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--val-interval", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--audio-root", type=str, default="", help="Root for audio files")
    return p.parse_args()


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ========================================================================
# Dataset
# ========================================================================
class S2MelDataset(Dataset):
    """Each sample: audio → mel (target) + semantic codes → length_regulator → cond (input)."""

    def __init__(self, manifest_paths, mel_cfg, audio_roots=None):
        self.samples = []
        self.mel_cfg = mel_cfg
        self.audio_roots = audio_roots or []

        for mp in manifest_paths:
            mp = Path(mp)
            base_dir = mp.parent
            log.info(f"[Info] Parsing manifest {mp} ...")
            count = skipped = 0
            with open(mp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1; continue

                    codes_path = self._resolve(base_dir, rec.get("codes_path", ""))
                    if not codes_path:
                        skipped += 1; continue

                    audio_path = self._resolve_audio(base_dir, rec.get("audio_path", ""))
                    if not audio_path:
                        skipped += 1; continue

                    self.samples.append({
                        "id": rec.get("id", ""),
                        "audio_path": str(audio_path),
                        "codes_path": str(codes_path),
                    })
                    count += 1
            log.info(f"  Loaded {count} samples (skipped {skipped}) from {mp.name}")
        log.info(f"[Info] Total S2Mel samples: {len(self.samples)}")

    def _resolve(self, base, p):
        if not p: return None
        pa = Path(p)
        if pa.is_absolute() and pa.exists(): return pa
        r = base / p
        return r if r.exists() else None

    def _resolve_audio(self, base, audio_str):
        if not audio_str: return None
        p = Path(audio_str)
        if p.is_absolute() and p.exists(): return p
        r = base / audio_str
        if r.exists(): return r
        for root in self.audio_roots:
            c = Path(root) / audio_str
            if c.exists(): return c
            c = Path(root) / p.name
            if c.exists(): return c
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            import torchaudio
            waveform, sr = torchaudio.load(s["audio_path"])
            waveform = waveform[0]  # mono
            # Resample to 22050 for mel extraction
            if sr != 22050:
                waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
            # Resample to 16000 for style extraction
            if sr != 16000:
                waveform_16k = torchaudio.transforms.Resample(sr, 16000)(torchaudio.load(s["audio_path"])[0][0])
            else:
                waveform_16k = waveform

            # Extract mel spectrogram at 22050Hz
            mel = mel_spectrogram(
                waveform.unsqueeze(0),
                **self.mel_cfg
            ).squeeze(0)  # (n_mels, T)

            # Extract fbank features for style (CAMPPlus input)
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
            )  # (T_fbank, 80)
            fbank = fbank - fbank.mean(dim=0, keepdim=True)

            # Load semantic codes
            codes = np.load(s["codes_path"])
            codes = torch.from_numpy(codes).long().squeeze()

            return {
                "mel": mel,          # (80, T_mel)
                "codes": codes,      # (T_codes,)
                "fbank": fbank,      # (T_fbank, 80)
            }
        except Exception as e:
            log.warning(f"  Error loading {s['id']}: {e}")
            return None


def collate_s2mel(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    mels = [b["mel"] for b in batch]
    codes_list = [b["codes"] for b in batch]
    fbanks = [b["fbank"] for b in batch]

    B = len(batch)
    n_mels = mels[0].size(0)
    max_mel = max(m.size(-1) for m in mels)
    max_code = max(c.size(0) for c in codes_list)
    max_fbank = max(f.size(0) for f in fbanks)

    mel_padded = torch.zeros(B, n_mels, max_mel)
    mel_lens = torch.zeros(B, dtype=torch.long)
    codes_padded = torch.zeros(B, max_code, dtype=torch.long)
    code_lens = torch.zeros(B, dtype=torch.long)
    fbank_padded = torch.zeros(B, max_fbank, 80)

    for i in range(B):
        T = mels[i].size(-1); mel_padded[i, :, :T] = mels[i]; mel_lens[i] = T
        L = codes_list[i].size(0); codes_padded[i, :L] = codes_list[i]; code_lens[i] = L
        F = fbanks[i].size(0); fbank_padded[i, :F, :] = fbanks[i]

    return {
        "mel": mel_padded, "mel_lens": mel_lens,
        "codes": codes_padded, "code_lens": code_lens,
        "fbank": fbank_padded,
    }


# ========================================================================
# Model
# ========================================================================
def build_model(cfg_path, base_checkpoint, device):
    """Build the full S2Mel model (MyModel with length_regulator + CFM)."""
    cfg = OmegaConf.load(cfg_path)

    model = MyModel(cfg.s2mel, use_gpt_latent=False)

    if base_checkpoint.exists():
        log.info(f"[Info] Loading checkpoint from {base_checkpoint}")
        model, _, _, _ = load_checkpoint2(model, None, base_checkpoint, is_distributed=False)
        log.info("[Info] S2Mel model loaded.")
    else:
        log.warning(f"[Warning] No checkpoint at {base_checkpoint}")

    # Also load CAMPPlus for style extraction
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)

    # Try loading campplus weights from base checkpoint
    if base_checkpoint.exists():
        state = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            cp_state = {}
            for k, v in state.items():
                if "campplus" in k or "style_encoder" in k:
                    clean_k = k.split("campplus.")[-1] if "campplus." in k else k
                    cp_state[clean_k] = v
            if cp_state:
                try:
                    campplus.load_state_dict(cp_state, strict=False)
                    log.info("[Info] CAMPPlus weights loaded from checkpoint")
                except:
                    log.warning("[Warning] Could not load CAMPPlus weights, using random init")

    return model.to(device), campplus.to(device), cfg


# ========================================================================
# Training
# ========================================================================
def compute_loss(model, campplus, batch, device, semantic_codec=None):
    """Compute S2Mel CFM loss.

    Pipeline:
    1. fbank → CAMPPlus → style (B, 192)
    2. codes → reshape for length_regulator → cond (B, T_mel, 512)
    3. cond + prompt_mel → CFM forward → loss
    """
    mel = batch["mel"].to(device)            # (B, 80, T_mel)
    mel_lens = batch["mel_lens"].to(device)  # (B,)
    codes = batch["codes"].to(device)        # (B, T_codes)
    code_lens = batch["code_lens"].to(device)
    fbank = batch["fbank"].to(device)        # (B, T_fbank, 80)

    B = mel.size(0)

    # 1. Extract style from fbank using CAMPPlus
    with torch.no_grad():
        style = campplus(fbank)  # (B, 192)

    # 2. Process codes through length_regulator to get cond
    # The length_regulator expects codes as (B, n_codebooks, T) format
    # and outputs (cond, _) where cond is (B, T_mel, 512)
    target_lengths = mel_lens  # target mel lengths
    codes_for_lr = codes.unsqueeze(1)  # (B, 1, T_codes)

    cond = model.models['length_regulator'](
        codes_for_lr, ylens=target_lengths, n_quantizers=1, f0=None
    )[0]  # (B, T_mel, 512)

    # 3. Prompt: use first ~20% of mel as prompt
    prompt_lens = (mel_lens.float() * 0.2).long().clamp(min=10)

    # 4. CFM forward loss
    loss, _ = model.models['cfm'](mel, mel_lens, prompt_lens, cond, style)
    return loss


def save_ckpt(path, model, optimizer, scheduler, scaler, epoch, step, recent):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch, "step": step,
    }, path)
    log.info(f"  Saved: {path}")
    recent.append(str(path))
    while len(recent) > 3:
        old = Path(recent.pop(0))
        if old.exists() and "latest" not in old.name:
            old.unlink()


def evaluate(model, campplus, loader, device):
    model.eval()
    total = n = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            try:
                loss = compute_loss(model, campplus, batch, device)
                total += loss.item(); n += 1
            except: continue
    model.train()
    return total / max(n, 1)


# ========================================================================
# Main
# ========================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Info] Device: {device}")

    cfg = OmegaConf.load(args.config)

    # Mel config for extraction
    mel_cfg = {
        "n_fft": cfg.s2mel.preprocess_params.spect_params.n_fft,
        "num_mels": cfg.s2mel.preprocess_params.spect_params.n_mels,
        "sampling_rate": cfg.s2mel.preprocess_params.sr,
        "hop_size": cfg.s2mel.preprocess_params.spect_params.hop_length,
        "win_size": cfg.s2mel.preprocess_params.spect_params.win_length,
        "fmin": cfg.s2mel.preprocess_params.spect_params.fmin,
        "fmax": None,
    }

    audio_roots = []
    if args.audio_root: audio_roots.append(args.audio_root)
    audio_roots.extend(["datasets/LJSpeech-1.1/wavs", "datasets/LJSpeech-1.1", "datasets"])

    log.info("[Info] Loading training manifests...")
    train_ds = S2MelDataset(args.train_manifests, mel_cfg, audio_roots)
    log.info("[Info] Loading validation manifests...")
    val_ds = S2MelDataset(args.val_manifests, mel_cfg, audio_roots)

    if len(train_ds) == 0:
        log.error("[Error] No training samples! Ensure LJSpeech audio is downloaded:")
        log.error("  wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2")
        log.error("  tar -xf LJSpeech-1.1.tar.bz2 -C datasets/")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_s2mel,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_s2mel,
                            pin_memory=True)

    log.info("[Info] Building S2Mel model...")
    model, campplus, full_cfg = build_model(args.config, args.base_checkpoint, device)
    campplus.eval()  # Style extractor is always in eval mode

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"[Info] Trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.grad_accumulation
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1; global_step = 0; recent: List[str] = []

    if args.resume:
        rp = None
        if args.resume == "auto":
            latest = args.output_dir / "latest.pth"
            if latest.exists(): rp = latest
        else:
            rp = Path(args.resume)
        if rp and rp.exists():
            log.info(f"[Info] Resuming from {rp}")
            ckpt = torch.load(rp, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
            if ckpt.get("scaler") and scaler: scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("step", 0)
            log.info(f"  Resumed: epoch {start_epoch}, step {global_step}")

    log.info(f"[Info] Training: {args.epochs} epochs, {len(train_loader)} batches/epoch")

    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        ep_loss = 0.0; ep_n = 0; t0 = time.time()

        for bi, batch in enumerate(train_loader):
            if batch is None: continue
            try:
                use_amp = args.amp and device.type == "cuda"
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = compute_loss(model, campplus, batch, device)
                    loss = loss / args.grad_accumulation

                if scaler: scaler.scale(loss).backward()
                else: loss.backward()

                if (bi + 1) % args.grad_accumulation == 0:
                    if scaler: scaler.unscale_(optimizer)
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    if scaler: scaler.step(optimizer); scaler.update()
                    else: optimizer.step()
                    scheduler.step(); optimizer.zero_grad()
                    global_step += 1

                    al = loss.item() * args.grad_accumulation
                    ep_loss += al; ep_n += 1

                    if global_step % args.log_interval == 0 or global_step == 1:
                        lr = optimizer.param_groups[0]["lr"]
                        log.info(f"[Train] epoch={epoch} step={global_step} loss={al:.4f} lr={lr:.2e}")

                    if args.val_interval > 0 and global_step % args.val_interval == 0:
                        vl = evaluate(model, campplus, val_loader, device)
                        log.info(f"[Val] step={global_step} loss={vl:.4f}")
                        save_ckpt(args.output_dir / f"model_step{global_step}.pth",
                                  model, optimizer, scheduler, scaler, epoch, global_step, recent)
                        save_ckpt(args.output_dir / "latest.pth",
                                  model, optimizer, scheduler, scaler, epoch, global_step, [])

                    if args.max_steps > 0 and global_step >= args.max_steps:
                        break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    log.warning(f"  OOM batch {bi}, skipping")
                    torch.cuda.empty_cache(); optimizer.zero_grad(); continue
                raise

        elapsed = time.time() - t0
        avg = ep_loss / max(ep_n, 1)
        log.info(f"[Epoch {epoch}/{args.epochs}] avg_loss={avg:.4f} time={elapsed:.0f}s")
        save_ckpt(args.output_dir / f"model_epoch{epoch}.pth",
                  model, optimizer, scheduler, scaler, epoch, global_step, recent)
        save_ckpt(args.output_dir / "latest.pth",
                  model, optimizer, scheduler, scaler, epoch, global_step, [])

        if len(val_ds) > 0:
            vl = evaluate(model, campplus, val_loader, device)
            log.info(f"[Val] epoch={epoch} loss={vl:.4f}")

    log.info("Training complete.")


if __name__ == "__main__":
    main()
