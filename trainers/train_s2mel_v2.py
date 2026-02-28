#!/usr/bin/env python3
"""
End-to-end finetuning entry point for IndexTTS2 S2Mel (DiT flow-matching) module.

The S2Mel model converts semantic tokens into Mel spectrograms using a
Conditional Flow Matching (CFM) approach with a DiT backbone.

This trainer loads preprocessed data manifests that contain paths to
pre-extracted features (semantic codes, conditioning embeddings, mel specs).
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

# ========================================================================
# Add project root to path
# ========================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from indextts.s2mel.modules.flow_matching import CFM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ========================================================================
# Mel extraction from audio (used when mel .npy not pre-saved)
# ========================================================================
def extract_mel_from_audio(audio_path: str, cfg) -> Optional[torch.Tensor]:
    """Extract mel spectrogram from an audio file.
    
    Returns:
        mel tensor of shape (n_mels, T) or None on failure
    """
    try:
        import torchaudio
        import torchaudio.transforms as T
    except ImportError:
        return None

    try:
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform[0]  # mono

        pp = cfg.s2mel.preprocess_params
        sp = pp.spect_params
        target_sr = pp.sr

        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=sp.n_fft,
            hop_length=sp.hop_length,
            win_length=sp.win_length,
            n_mels=sp.n_mels,
            f_min=sp.fmin,
            f_max=None,
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        )

        mel = mel_transform(waveform.unsqueeze(0))
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze(0)  # (n_mels, T)
    except Exception as e:
        log.warning(f"  Failed to extract mel from {audio_path}: {e}")
        return None


# ========================================================================
# CLI
# ========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune IndexTTS2 S2Mel (DiT) model.")
    parser.add_argument("--train-manifest", dest="train_manifests", action="append", type=str, required=True)
    parser.add_argument("--val-manifest", dest="val_manifests", action="append", type=str, required=True)
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/s2mel.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("trained_ckpts_s2mel"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--val-interval", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--audio-root", type=str, default="", help="Root directory for audio files if paths are relative")
    return parser.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========================================================================
# Dataset  
# ========================================================================
class S2MelDataset(Dataset):
    """Dataset for S2Mel training.
    
    Loads pre-computed features from the preprocessing pipeline:
    - semantic codes (.npy) - used for conditioning input (mu)
    - conditioning embeddings (.npy) - w2v-bert style features
    - mel spectrograms - extracted from audio on-the-fly
    """

    def __init__(self, manifest_paths: Sequence[str], cfg, audio_roots: Optional[List[str]] = None):
        self.samples = []
        self.cfg = cfg
        self.audio_roots = audio_roots or []

        for mp in manifest_paths:
            mp = Path(mp)
            base_dir = mp.parent
            log.info(f"[Info] Parsing manifest {mp} ...")
            count = 0
            skipped = 0
            with open(mp, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    # Resolve feature paths relative to manifest dir
                    codes_path = self._resolve_path(base_dir, record.get("codes_path", ""))
                    condition_path = self._resolve_path(base_dir, record.get("condition_path", ""))
                    emo_vec_path = self._resolve_path(base_dir, record.get("emo_vec_path", ""))

                    # Check required features exist
                    if not codes_path or not codes_path.exists():
                        skipped += 1
                        continue
                    if not condition_path or not condition_path.exists():
                        skipped += 1
                        continue

                    # Resolve audio path (try multiple roots)
                    audio_path = self._resolve_audio(base_dir, record.get("audio_path", ""))

                    # Check mel path (pre-computed)
                    mel_path = self._resolve_path(base_dir, record.get("mel_path", ""))

                    if not audio_path and not (mel_path and mel_path.exists()):
                        skipped += 1
                        continue

                    self.samples.append({
                        "id": record.get("id", f"sample_{line_num}"),
                        "audio_path": str(audio_path) if audio_path else None,
                        "mel_path": str(mel_path) if mel_path and mel_path.exists() else None,
                        "codes_path": str(codes_path),
                        "condition_path": str(condition_path),
                        "emo_vec_path": str(emo_vec_path) if emo_vec_path and emo_vec_path.exists() else None,
                        "code_len": record.get("code_len", 0),
                        "condition_len": record.get("condition_len", 0),
                    })
                    count += 1

            log.info(f"  Loaded {count} samples (skipped {skipped}) from {mp.name}")
        log.info(f"[Info] Total S2Mel training samples: {len(self.samples)}")

    def _resolve_path(self, base_dir: Path, path_str: str) -> Optional[Path]:
        if not path_str:
            return None
        p = Path(path_str)
        if p.is_absolute() and p.exists():
            return p
        resolved = base_dir / path_str
        if resolved.exists():
            return resolved
        return None

    def _resolve_audio(self, base_dir: Path, audio_str: str) -> Optional[Path]:
        if not audio_str:
            return None
        # Try absolute
        p = Path(audio_str)
        if p.is_absolute() and p.exists():
            return p
        # Try relative to manifest dir
        resolved = base_dir / audio_str
        if resolved.exists():
            return resolved
        # Try each audio root
        for root in self.audio_roots:
            candidate = Path(root) / audio_str
            if candidate.exists():
                return candidate
            # Try just the filename
            candidate = Path(root) / Path(audio_str).name
            if candidate.exists():
                return candidate
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        sample = self.samples[idx]

        # Load semantic codes
        codes = np.load(sample["codes_path"])
        codes = torch.from_numpy(codes).long().squeeze()

        # Load condition embedding (w2v-bert features)
        cond = np.load(sample["condition_path"])
        cond = torch.from_numpy(cond).float()  # (cond_len, dim)

        # Load emo vector if available
        if sample["emo_vec_path"]:
            emo = np.load(sample["emo_vec_path"])
            emo = torch.from_numpy(emo).float().squeeze()
        else:
            emo = torch.zeros(192)  # default style dim

        # Load mel - prefer pre-computed, else extract from audio
        mel = None
        if sample["mel_path"]:
            mel_np = np.load(sample["mel_path"])
            mel = torch.from_numpy(mel_np).float()
        elif sample["audio_path"]:
            mel = extract_mel_from_audio(sample["audio_path"], self.cfg)

        if mel is None:
            # Return a dummy that will be filtered in collate
            return None

        # Ensure mel is (n_mels, T)
        if mel.dim() == 3:
            mel = mel.squeeze(0)

        return {
            "mel": mel,
            "codes": codes,
            "condition": cond,
            "emo": emo,
        }


def collate_s2mel(batch):
    """Collate function that filters None samples."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    mels = [item["mel"] for item in batch]
    codes_list = [item["codes"] for item in batch]
    conditions = [item["condition"] for item in batch]
    emos = [item["emo"] for item in batch]

    max_mel_len = max(m.size(-1) for m in mels)
    max_cond_len = max(c.size(0) for c in conditions)
    n_mels = mels[0].size(0)
    cond_dim = conditions[0].size(-1)
    B = len(batch)

    mel_padded = torch.zeros(B, n_mels, max_mel_len)
    mel_lens = torch.zeros(B, dtype=torch.long)
    for i, m in enumerate(mels):
        T = m.size(-1)
        mel_padded[i, :, :T] = m
        mel_lens[i] = T

    cond_padded = torch.zeros(B, max_cond_len, cond_dim)
    cond_lens = torch.zeros(B, dtype=torch.long)
    for i, c in enumerate(conditions):
        L = c.size(0)
        cond_padded[i, :L, :] = c
        cond_lens[i] = L

    emo_stacked = torch.stack(emos, dim=0)

    return {
        "mel": mel_padded,
        "mel_lens": mel_lens,
        "condition": cond_padded,
        "cond_lens": cond_lens,
        "emo": emo_stacked,
    }


# ========================================================================
# Model
# ========================================================================
def build_s2mel_model(cfg_path: Path, base_checkpoint: Path, device: torch.device) -> CFM:
    cfg = OmegaConf.load(cfg_path)
    model = CFM(cfg.s2mel)

    if base_checkpoint.exists():
        log.info(f"[Info] Loading base S2Mel checkpoint from {base_checkpoint}")
        state_dict = torch.load(base_checkpoint, map_location="cpu", weights_only=True)

        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("s2mel."):
                cleaned[k[6:]] = v
            elif k.startswith("module."):
                cleaned[k[7:]] = v
            else:
                cleaned[k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            log.warning(f"  Missing keys: {len(missing)} - {missing[:5]}")
        if unexpected:
            log.warning(f"  Unexpected keys: {len(unexpected)} - {unexpected[:5]}")
        log.info("[Info] S2Mel model loaded successfully.")
    else:
        log.warning(f"[Warning] No base checkpoint at {base_checkpoint}, training from scratch.")

    return model.to(device)


# ========================================================================
# Training
# ========================================================================
def compute_s2mel_loss(model, batch, device, style_dim=192):
    mel = batch["mel"].to(device)
    mel_lens = batch["mel_lens"].to(device)
    condition = batch["condition"].to(device)
    cond_lens = batch["cond_lens"]
    emo = batch["emo"].to(device)

    B = mel.size(0)
    max_mel_len = mel.size(-1)
    cond_dim = condition.size(-1)

    # Prompt = first ~20% of mel as reference
    prompt_lens = (mel_lens.float() * 0.2).long().clamp(min=10)

    # Build mu: pad/truncate condition to match mel time dimension
    # condition is (B, cond_T, dim), need (B, mel_T, dim)
    mu = torch.zeros(B, max_mel_len, cond_dim, device=device)
    for i in range(B):
        clen = min(cond_lens[i].item(), max_mel_len)
        mu[i, :clen, :] = condition[i, :clen, :]

    # Style vector
    if emo.size(-1) != style_dim:
        style = emo[:, :style_dim] if emo.size(-1) > style_dim else torch.zeros(B, style_dim, device=device)
    else:
        style = emo

    loss, _ = model(mel, mel_lens, prompt_lens, mu, style)
    return loss


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, step, recent_checkpoints):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
    }
    torch.save(state, path)
    log.info(f"  Saved checkpoint: {path}")

    recent_checkpoints.append(str(path))
    while len(recent_checkpoints) > 3:
        old = recent_checkpoints.pop(0)
        old_path = Path(old)
        if old_path.exists() and "latest" not in old_path.name:
            old_path.unlink()


def evaluate(model, loader, device, style_dim=192):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            try:
                loss = compute_s2mel_loss(model, batch, device, style_dim)
                total_loss += loss.item()
                n += 1
            except Exception as e:
                log.warning(f"  Val batch error: {e}")
                continue
    model.train()
    return total_loss / max(n, 1)


# ========================================================================
# Main
# ========================================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Info] Device: {device}")

    cfg = OmegaConf.load(args.config)

    # Determine style dim from config
    style_dim = cfg.s2mel.get("style_encoder", {}).get("dim", 192)

    # Audio roots for resolving audio paths
    audio_roots = []
    if args.audio_root:
        audio_roots.append(args.audio_root)
    # Common locations
    audio_roots.extend(["datasets/LJSpeech-1.1/wavs", "datasets/LJSpeech-1.1", "datasets"])

    log.info("[Info] Loading training manifests...")
    train_dataset = S2MelDataset(args.train_manifests, cfg, audio_roots)
    log.info("[Info] Loading validation manifests...")
    val_dataset = S2MelDataset(args.val_manifests, cfg, audio_roots)

    if len(train_dataset) == 0:
        log.error("[Error] No training samples loaded! Check that audio files exist.")
        log.error("  Audio roots searched: " + str(audio_roots))
        log.error("  Try downloading LJSpeech first:")
        log.error("    wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2")
        log.error("    tar -xf LJSpeech-1.1.tar.bz2 -C datasets/")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_s2mel,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_s2mel,
        pin_memory=True,
    )

    log.info("[Info] Building S2Mel model...")
    model = build_s2mel_model(args.config, args.base_checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"[Info] Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // args.grad_accumulation
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)

    start_epoch = 1
    global_step = 0
    recent_checkpoints: List[str] = []

    if args.resume:
        resume_path = None
        if args.resume == "auto":
            latest = args.output_dir / "latest.pth"
            if latest.exists():
                resume_path = latest
        else:
            resume_path = Path(args.resume)

        if resume_path and resume_path.exists():
            log.info(f"[Info] Resuming from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scheduler") and scheduler:
                scheduler.load_state_dict(ckpt["scheduler"])
            if ckpt.get("scaler") and scaler:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("step", 0)
            log.info(f"  Resumed at epoch {start_epoch}, step {global_step}")

    log.info(f"[Info] Starting training: {args.epochs} epochs, {len(train_loader)} batches/epoch")

    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        epoch_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            try:
                use_amp = args.amp and device.type == "cuda"
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = compute_s2mel_loss(model, batch, device, style_dim)
                    loss = loss / args.grad_accumulation

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % args.grad_accumulation == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    actual_loss = loss.item() * args.grad_accumulation
                    epoch_loss += actual_loss
                    epoch_batches += 1

                    if global_step % args.log_interval == 0 or global_step == 1:
                        lr = optimizer.param_groups[0]["lr"]
                        log.info(f"[Train] epoch={epoch} step={global_step} loss={actual_loss:.4f} lr={lr:.2e}")

                    if args.val_interval > 0 and global_step % args.val_interval == 0:
                        val_loss = evaluate(model, val_loader, device, style_dim)
                        log.info(f"[Val] epoch={epoch} step={global_step} loss={val_loss:.4f}")
                        ckpt_path = args.output_dir / f"model_step{global_step}.pth"
                        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, epoch, global_step, recent_checkpoints)
                        latest_path = args.output_dir / "latest.pth"
                        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, global_step, [])

                    if args.max_steps > 0 and global_step >= args.max_steps:
                        log.info(f"[Info] Reached max steps {args.max_steps}")
                        break

            except RuntimeError as e:
                if "out of memory" in str(e):
                    log.warning(f"  OOM at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(epoch_batches, 1)
        log.info(f"[Epoch {epoch}/{args.epochs}] avg_loss={avg_loss:.4f} time={elapsed:.0f}s")

        ckpt_path = args.output_dir / f"model_epoch{epoch}.pth"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, epoch, global_step, recent_checkpoints)
        latest_path = args.output_dir / "latest.pth"
        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, global_step, [])

        if len(val_dataset) > 0:
            val_loss = evaluate(model, val_loader, device, style_dim)
            log.info(f"[Val] epoch={epoch} loss={val_loss:.4f}")

    log.info("Training complete.")


if __name__ == "__main__":
    main()
