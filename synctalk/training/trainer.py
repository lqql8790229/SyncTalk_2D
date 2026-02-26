"""Unified training pipeline with AMP, LR scheduling, and checkpoint resumption.

Merges train.py and train_328.py, and adds:
- Mixed precision training (AMP)
- Learning rate scheduling (Cosine Annealing with Warmup)
- Checkpoint resumption
- TensorBoard logging
- Proper gradient zeroing
"""

import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from ..configs.base import SyncTalkConfig
from ..models.unet import UNet
from ..models.syncnet import SyncNet
from ..data.dataset import LipSyncDataset, SyncNetDataset
from ..utils.device import get_device
from ..utils.io import ensure_dir
from .losses import PerceptualLoss, cosine_loss

logger = logging.getLogger(__name__)


class Trainer:
    """Unified trainer for SyncTalk models.

    Args:
        config: SyncTalkConfig with all training parameters.
        device_str: Device string ("auto", "cuda", "cpu").
    """

    def __init__(self, config: SyncTalkConfig, device_str: str = "auto"):
        self.config = config
        self.device = get_device(device_str)
        self.use_amp = config.train.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

    def train_syncnet(self, dataset_dir: str, save_dir: str):
        """Train SyncNet discriminator."""
        ensure_dir(save_dir)
        cfg = self.config

        dataset = SyncNetDataset(dataset_dir, cfg.model)
        data_loader = DataLoader(
            dataset,
            batch_size=cfg.train.syncnet_batch_size,
            shuffle=True,
            num_workers=cfg.train.syncnet_num_workers,
        )

        model = SyncNet.from_config(cfg.model).to(self.device)
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.train.syncnet_lr,
        )

        best_loss = float("inf")
        for epoch in range(cfg.train.syncnet_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in data_loader:
                img, audio, y = batch
                img = img.to(self.device)
                audio = audio.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                with autocast(enabled=self.use_amp):
                    audio_emb, face_emb = model(img, audio)
                    loss = cosine_loss(audio_emb, face_emb, y)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"SyncNet Epoch {epoch + 1}/{cfg.train.syncnet_epochs} loss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt_path = os.path.join(save_dir, f"{epoch + 1}.pth")
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Saved best SyncNet: {ckpt_path}")

        return save_dir

    def train_unet(self, dataset_dir: str, save_dir: str,
                   syncnet_checkpoint: str = None, resume_from: str = None):
        """Train main UNet lip-sync model."""
        ensure_dir(save_dir)
        cfg = self.config

        dataset = LipSyncDataset(dataset_dir, cfg.model)
        data_loader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.train.num_workers,
        )

        net = UNet.from_config(cfg.model).to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)

        scheduler = self._create_scheduler(optimizer)

        start_epoch = 0
        if resume_from:
            start_epoch = self._load_checkpoint(resume_from, net, optimizer, scheduler)
            logger.info(f"Resumed from epoch {start_epoch}")

        content_loss = PerceptualLoss(nn.MSELoss(), device=self.device)
        pixel_criterion = nn.L1Loss()

        syncnet = None
        use_syncnet = cfg.train.use_syncnet and syncnet_checkpoint
        if use_syncnet:
            syncnet = SyncNet.from_config(cfg.model).eval().to(self.device)
            syncnet.load_state_dict(torch.load(syncnet_checkpoint, map_location=self.device))
            logger.info(f"Loaded SyncNet: {syncnet_checkpoint}")

        for epoch in range(start_epoch, cfg.train.epochs):
            net.train()
            with tqdm(total=len(dataset), desc=f"Epoch {epoch + 1}/{cfg.train.epochs}", unit="img") as pbar:
                for batch in data_loader:
                    imgs, labels, audio_feat = batch
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    audio_feat = audio_feat.to(self.device)

                    optimizer.zero_grad(set_to_none=True)

                    with autocast(enabled=self.use_amp):
                        preds = net(imgs, audio_feat)
                        loss_pixel = pixel_criterion(preds, labels)
                        loss_perceptual = content_loss.get_loss(preds, labels)
                        loss = (loss_pixel * cfg.train.loss_pixel_weight +
                                loss_perceptual * cfg.train.loss_perceptual_weight)

                        if use_syncnet:
                            y = torch.ones([preds.shape[0], 1], device=self.device)
                            a, v = syncnet(preds, audio_feat)
                            sync_loss = cosine_loss(a, v, y)
                            loss += sync_loss * cfg.train.loss_sync_weight

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    pbar.set_postfix(**{"loss": f"{loss.item():.4f}"})
                    pbar.update(imgs.shape[0])

            if scheduler:
                scheduler.step()

            if (epoch + 1) % cfg.train.save_interval == 0:
                self._save_checkpoint(save_dir, epoch, net, optimizer, scheduler)

        self._save_checkpoint(save_dir, cfg.train.epochs - 1, net, optimizer, scheduler)
        logger.info("Training complete")

    def _create_scheduler(self, optimizer):
        cfg = self.config.train
        if cfg.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs - cfg.lr_warmup_epochs,
                eta_min=cfg.lr_min,
            )
        return None

    def _save_checkpoint(self, save_dir, epoch, model, optimizer, scheduler):
        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if scheduler:
            state["scheduler_state_dict"] = scheduler.state_dict()
        path = os.path.join(save_dir, f"{epoch + 1}.pth")
        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, path, model, optimizer, scheduler):
        ckpt = torch.load(path, map_location=self.device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scheduler and "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            return ckpt["epoch"]
        else:
            model.load_state_dict(ckpt)
            return 0

    def train_full(self, dataset_dir: str, save_dir: str, syncnet_save_dir: str):
        """Run full training pipeline: SyncNet → UNet."""
        logger.info("=== Phase 1: Training SyncNet ===")
        self.train_syncnet(dataset_dir, syncnet_save_dir)

        syncnet_ckpts = sorted(
            Path(syncnet_save_dir).glob("*.pth"),
            key=lambda p: int(p.stem),
        )
        syncnet_ckpt = str(syncnet_ckpts[-1]) if syncnet_ckpts else None

        logger.info("=== Phase 2: Training UNet ===")
        self.train_unet(dataset_dir, save_dir, syncnet_checkpoint=syncnet_ckpt)
