#!/usr/bin/env python3
"""
End-to-end finetuning entry point for IndexTTS2 S2Mel (DiT flow-matching) module.

The S2Mel model converts semantic tokens into Mel spectrograms using a
Conditional Flow Matching (CFM) approach with a DiT backbone.

This trainer loads preprocessed data (audio + semantic codes + condition embeddings)
and trains the DiT model to predict mel spectrograms from semantic representations.
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune IndexTTS2 S2Mel (DiT) model.")
    parser.add_argument(
        "--train-manifest",
        dest="train_manifests",
        action="append",
        type=str,
        required=True,
        help="Training manifest JSONL.",
    )
    parser.add_argument(
        "--val-manifest",
        dest="val_manifests",
        action="append",
        type=str,
        required=True,
        help="Validation manifest JSONL.",
    )
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"), help="Model config YAML.")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/s2mel.pth"), help="Base S2Mel checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("trained_ckpts_s2mel"), help="Directory for checkpoints/logs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--grad-accumulation", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=500, help="LR warmup steps.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max optimiser steps (0 = unlimited).")
    parser.add_argument("--log-interval", type=int, default=5, help="Steps between training log entries.")
    parser.add_argument("--val-interval", type=int, default=0, help="Validation frequency in steps (0 = once per epoch).")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping value.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from, or 'auto'.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========================================================================
# Mel extraction utility
# ========================================================================
class MelExtractor:
    """Extract mel spectrograms from audio waveforms."""

    def __init__(self, cfg):
        pp = cfg.s2mel.preprocess_params
        sp = pp.spect_params
        self.sr = pp.sr
        self.n_fft = sp.n_fft
        self.hop_length = sp.hop_length
        self.win_length = sp.win_length
        self.n_mels = sp.n_mels
        self.fmin = sp.fmin
        fmax = sp.get("fmax", None)
        self.fmax = None if fmax == "None" or fmax is None else float(fmax)

    def extract(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract mel spectrogram from waveform tensor.
        
        Args:
            waveform: (num_samples,) audio tensor
            sr: sample rate
            
        Returns:
            mel: (n_mels, T) mel spectrogram
        """
        import torchaudio
        import torchaudio.transforms as T

        # Resample if needed
        if sr != self.sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sr)
            waveform = resampler(waveform)

        # Compute mel spectrogram
        mel_transform = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        )

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze(0)  # (n_mels, T)


# ========================================================================
# Dataset
# ========================================================================
@dataclass
class S2MelSample:
    id: str
    audio_path: str
    codes_path: Path
    condition_path: Path
    emo_vec_path: Path
    code_len: int
    condition_len: int
    manifest_path: Optional[Path] = None


class S2MelDataset(Dataset):
    """Dataset for S2Mel training.
    
    Each sample consists of:
    - mel spectrogram (extracted on-the-fly from audio)
    - semantic codes (loaded from preprocessed .npy)
    - conditioning embedding (loaded from preprocessed .npy)
    - emo vector (loaded from preprocessed .npy)
    """

    def __init__(self, manifest_paths: Sequence[str], mel_extractor: MelExtractor):
        self.samples: List[S2MelSample] = []
        self.mel_extractor = mel_extractor

        for mp in manifest_paths:
            mp = Path(mp)
            base_dir = mp.parent
            log.info(f"[Info] Parsing manifest {mp} ...")
            count = 0
            with open(mp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)

                    # Resolve paths relative to manifest
                    codes_path = base_dir / record["codes_path"] if not Path(record["codes_path"]).is_absolute() else Path(record["codes_path"])
                    condition_path = base_dir / record["condition_path"] if not Path(record["condition_path"]).is_absolute() else Path(record["condition_path"])
                    emo_vec_path = base_dir / record["emo_vec_path"] if not Path(record["emo_vec_path"]).is_absolute() else Path(record["emo_vec_path"])

                    # Resolve audio path
                    audio_path = record.get("audio_path", "")
                    if audio_path and not Path(audio_path).is_absolute():
                        audio_path = str(base_dir / audio_path)

                    if not Path(audio_path).exists():
                        continue
                    if not codes_path.exists():
                        continue

                    self.samples.append(S2MelSample(
                        id=record["id"],
                        audio_path=audio_path,
                        codes_path=codes_path,
                        condition_path=condition_path,
                        emo_vec_path=emo_vec_path,
                        code_len=record.get("code_len", 0),
                        condition_len=record.get("condition_len", 0),
                        manifest_path=mp,
                    ))
                    count += 1

            log.info(f"    Loaded {count} samples from {mp.name}")
        log.info(f"[Info] Total S2Mel training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load semantic codes
        codes = np.load(sample.codes_path)
        codes = torch.from_numpy(codes).long().squeeze()  # (code_len,)

        # Load condition embedding
        cond = np.load(sample.condition_path)
        cond = torch.from_numpy(cond).float()  # (cond_len, dim)

        # Load emo vector
        emo = np.load(sample.emo_vec_path)
        emo = torch.from_numpy(emo).float().squeeze()  # (emo_dim,)

        # Load audio and extract mel
        import torchaudio
        waveform, sr = torchaudio.load(sample.audio_path)
        waveform = waveform[0]  # mono
        mel = self.mel_extractor.extract(waveform, sr)  # (n_mels, T)

        # We need to match mel length with code length
        # The semantic codes have a different temporal resolution than mel
        # We'll handle alignment in the collate function

        return {
            "mel": mel,           # (n_mels, T)
            "codes": codes,       # (code_len,)
            "condition": cond,    # (cond_len, dim)
            "emo": emo,           # (emo_dim,)
            "id": sample.id,
        }


def collate_s2mel(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for S2Mel batches."""
    mels = [item["mel"] for item in batch]
    codes_list = [item["codes"] for item in batch]
    conditions = [item["condition"] for item in batch]
    emos = [item["emo"] for item in batch]

    # Get max lengths
    max_mel_len = max(m.size(-1) for m in mels)
    max_code_len = max(c.size(0) for c in codes_list)
    max_cond_len = max(c.size(0) for c in conditions)
    n_mels = mels[0].size(0)
    cond_dim = conditions[0].size(-1)

    B = len(batch)

    # Pad mels: (B, n_mels, T)
    mel_padded = torch.zeros(B, n_mels, max_mel_len)
    mel_lens = torch.zeros(B, dtype=torch.long)
    for i, m in enumerate(mels):
        T = m.size(-1)
        mel_padded[i, :, :T] = m
        mel_lens[i] = T

    # Pad codes: (B, max_code_len)
    codes_padded = torch.zeros(B, max_code_len, dtype=torch.long)
    code_lens = torch.zeros(B, dtype=torch.long)
    for i, c in enumerate(codes_list):
        L = c.size(0)
        codes_padded[i, :L] = c
        code_lens[i] = L

    # Pad conditions: (B, max_cond_len, dim)
    cond_padded = torch.zeros(B, max_cond_len, cond_dim)
    cond_lens = torch.zeros(B, dtype=torch.long)
    for i, c in enumerate(conditions):
        L = c.size(0)
        cond_padded[i, :L, :] = c
        cond_lens[i] = L

    # Stack emos: (B, emo_dim)
    emo_stacked = torch.stack(emos, dim=0)

    return {
        "mel": mel_padded,
        "mel_lens": mel_lens,
        "codes": codes_padded,
        "code_lens": code_lens,
        "condition": cond_padded,
        "cond_lens": cond_lens,
        "emo": emo_stacked,
    }


# ========================================================================
# Model building
# ========================================================================
def build_s2mel_model(cfg_path: Path, base_checkpoint: Path, device: torch.device) -> CFM:
    """Build and load the S2Mel CFM model."""
    cfg = OmegaConf.load(cfg_path)
    model = CFM(cfg.s2mel)
    
    if base_checkpoint.exists():
        log.info(f"[Info] Loading base S2Mel checkpoint from {base_checkpoint}")
        state_dict = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        
        # Handle different checkpoint formats
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # Remove prefix if present
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
            log.warning(f"  Missing keys: {len(missing)}")
        if unexpected:
            log.warning(f"  Unexpected keys: {len(unexpected)}")
        log.info(f"[Info] S2Mel model loaded successfully.")
    else:
        log.warning(f"[Warning] No base checkpoint found at {base_checkpoint}, training from scratch.")

    return model.to(device)


# ========================================================================
# Training utilities
# ========================================================================
def compute_s2mel_loss(
    model: CFM,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Compute the CFM flow matching loss."""
    mel = batch["mel"].to(device)          # (B, n_mels, T)
    mel_lens = batch["mel_lens"].to(device)  # (B,)
    condition = batch["condition"].to(device)  # (B, cond_len, dim)
    emo = batch["emo"].to(device)  # (B, emo_dim)

    # Use a portion of the mel as prompt (e.g., first 20%)
    B = mel.size(0)
    prompt_lens = (mel_lens.float() * 0.2).long().clamp(min=10)

    # Condition (semantic features) needs to be transposed for the model: (B, T, dim) -> (B, dim, T)
    # The model expects mu as (B, T, dim) format based on flow_matching.py forward
    # Pad/truncate condition to match mel length
    max_mel_len = mel.size(-1)
    cond_dim = condition.size(-1)
    mu = torch.zeros(B, max_mel_len, cond_dim, device=device)
    for i in range(B):
        cond_len = min(batch["cond_lens"][i].item(), max_mel_len)
        mu[i, :cond_len, :] = condition[i, :cond_len, :]

    # Style from emo vector - adapt to expected dim
    style_dim = 192  # From config_light.yaml style_encoder.dim
    if emo.size(-1) != style_dim:
        # Simple projection or truncation
        if emo.size(-1) > style_dim:
            style = emo[:, :style_dim]
        else:
            style = torch.zeros(B, style_dim, device=device)
            style[:, :emo.size(-1)] = emo
    else:
        style = emo

    loss, _ = model(mel, mel_lens, prompt_lens, mu, style)
    return loss


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    recent_checkpoints: List[str],
):
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

    # Keep maximum 3 recent checkpoints
    recent_checkpoints.append(str(path))
    while len(recent_checkpoints) > 3:
        old = recent_checkpoints.pop(0)
        old_path = Path(old)
        if old_path.exists() and "latest" not in old_path.name:
            old_path.unlink()
            log.info(f"  Removed old checkpoint: {old}")


def evaluate(
    model: CFM,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            try:
                loss = compute_s2mel_loss(model, batch, device)
                total_loss += loss.item()
                n_batches += 1
            except Exception as e:
                log.warning(f"  Validation batch error: {e}")
                continue
    model.train()
    return total_loss / max(n_batches, 1)


# ========================================================================
# Main training loop
# ========================================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Info] Device: {device}")

    # Load config
    cfg = OmegaConf.load(args.config)

    # Build mel extractor
    mel_extractor = MelExtractor(cfg)

    # Build datasets
    log.info("[Info] Loading training manifests...")
    train_dataset = S2MelDataset(args.train_manifests, mel_extractor)
    log.info("[Info] Loading validation manifests...")
    val_dataset = S2MelDataset(args.val_manifests, mel_extractor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_s2mel,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_s2mel,
        pin_memory=True,
    )

    # Build model
    log.info("[Info] Building S2Mel model...")
    model = build_s2mel_model(args.config, args.base_checkpoint, device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"[Info] Trainable parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accumulation
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)

    # Resume
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
        else:
            log.info("[Info] No checkpoint found for resume, starting fresh.")

    # Training loop
    log.info(f"[Info] Starting training: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    log.info(f"[Info] Effective batch size: {args.batch_size * args.grad_accumulation}")

    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        epoch_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            try:
                use_amp = args.amp and device.type == "cuda"
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = compute_s2mel_loss(model, batch, device)
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

                    # Log
                    if global_step % args.log_interval == 0 or global_step == 1:
                        lr = optimizer.param_groups[0]["lr"]
                        log.info(
                            f"[Train] epoch={epoch} step={global_step} "
                            f"loss={actual_loss:.4f} "
                            f"lr={lr:.2e}"
                        )

                    # Validation
                    if args.val_interval > 0 and global_step % args.val_interval == 0:
                        val_loss = evaluate(model, val_loader, device)
                        log.info(
                            f"[Val] epoch={epoch} step={global_step} "
                            f"loss={val_loss:.4f}"
                        )

                    # Save checkpoint
                    if args.val_interval > 0 and global_step % args.val_interval == 0:
                        ckpt_path = args.output_dir / f"model_step{global_step}.pth"
                        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, epoch, global_step, recent_checkpoints)
                        # Always save latest
                        latest_path = args.output_dir / "latest.pth"
                        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, global_step, [])

                    # Max steps
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

        # End of epoch
        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(epoch_batches, 1)
        log.info(
            f"[Epoch {epoch}/{args.epochs}] avg_loss={avg_loss:.4f} "
            f"time={elapsed:.0f}s"
        )

        # Save epoch checkpoint
        ckpt_path = args.output_dir / f"model_epoch{epoch}.pth"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, epoch, global_step, recent_checkpoints)
        latest_path = args.output_dir / "latest.pth"
        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, global_step, [])

        # Epoch validation
        if val_loader and len(val_dataset) > 0:
            val_loss = evaluate(model, val_loader, device)
            log.info(f"[Val] epoch={epoch} loss={val_loss:.4f}")

    log.info("Training complete.")


if __name__ == "__main__":
    main()
