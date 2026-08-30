"""Content Extractor Model — ocean.Model for Emformer training.

Trains the Stream Content Extractor to regress to HuBERT's continuous
256-dim content embeddings via MSE loss (no clustering needed).

Training: 80k steps, Adam (beta1=0.9, beta2=0.98)
"""

import logging

import paddle
import paddle.nn.functional as F

from ocean.model import Model

from layers.dataset import ContentExtractorDataset
from layers.stream_content_extractor import StreamContentExtractor
from utils.training_utils import build_train_dataloader, build_val_dataloader

logger = logging.getLogger(__name__)


class ContentExtractorModel(Model):
    """Content Extractor training model.

    Distills HuBERT's continuous 256-dim content embeddings into the
    streaming Emformer encoder via MSE regression.
    No clustering required — matches the approach verified in SVC4.

    Args:
        config: Training configuration dict.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.automatic_optimization = True
        self.skip_immediate_validation = False

        sce_cfg = config.get("content_extractor", {})
        audio_cfg = config.get("audio", {})

        self.extractor = StreamContentExtractor(
            input_dim=audio_cfg.get("num_mels", 80),
            d_model=sce_cfg.get("d_model", 512),
            nhead=sce_cfg.get("nhead", 8),
            num_layers=sce_cfg.get("num_layers", 6),
            output_dim=sce_cfg.get("output_dim", 256),
            chunk_size=sce_cfg.get("chunk_size", 4),
            left_context=sce_cfg.get("left_context", 1),
            right_context=sce_cfg.get("right_context", 2),
            dim_feedforward=sce_cfg.get("dim_feedforward", 2048),
            dropout=sce_cfg.get("dropout", 0.1),
        )

    def forward(self, mel: paddle.Tensor) -> paddle.Tensor:
        """Forward pass.

        Args:
            mel: (B, n_mels, T) mel-spectrogram.

        Returns:
            content_emb: (B, T, 256) content embeddings.
        """
        return self.extractor(mel.transpose([0, 2, 1]))

    def training_step(self, batch: dict, batch_idx: int) -> paddle.Tensor:
        """Training step with MSE regression loss.

        Args:
            batch: Dict with ``source_mel`` (B, n_mels, T) and
                   ``hubert_emb`` (B, T, 256) HuBERT continuous embeddings.

        Returns:
            loss.
        """
        mel = batch["source_mel"]
        hubert_emb = batch["hubert_emb"]
        content_emb = self.extractor(mel.transpose([0, 2, 1]))
        T_pred = content_emb.shape[1]
        T_target = hubert_emb.shape[1]
        if T_pred > T_target:
            content_emb = content_emb[:, :T_target, :]
        elif T_pred < T_target:
            content_emb = F.pad(content_emb, [0, 0, 0, T_target - T_pred])
        valid_mask = batch["valid_mask"].unsqueeze(-1)
        squared_error = ((content_emb - hubert_emb) ** 2) * valid_mask
        loss = squared_error.sum() / paddle.clip(valid_mask.sum() * content_emb.shape[-1], min=1.0)

        # Progress bar only (DiffSinger: logger=False for tqdm)
        self.log("train/loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False)

        # Direct logger write at log_interval boundaries — only on last accumulation batch
        log_every = self.config.get("training", {}).get("log_every", 100)
        if self.global_step % log_every == 0:
            if self.logger is not None:
                # Get current LR
                lr_val = 0.0
                if self._trainer and self._trainer.optimizers:
                    opt = self._trainer.optimizers[0]._optimizer
                    lr = getattr(opt, "_learning_rate", None)
                    if lr is not None:
                        if hasattr(lr, "get_lr"):
                            lr_val = lr.get_lr()
                        elif hasattr(lr, "numpy"):
                            lr_val = lr.numpy().item()
                        else:
                            lr_val = float(lr)
                acc = self._trainer.accumulate_grad_batches if self._trainer else 1
                if (batch_idx + 1) % acc == 0:
                    self.logger.log_metrics({
                        "train/loss": loss.item(),
                        "train/lr": lr_val,
                    }, step=self.global_step)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step — skip sanity check, Ocean auto-reduces self.log()."""
        if self.skip_immediate_validation:
            return
        mel = batch["source_mel"]
        hubert_emb = batch["hubert_emb"]
        content_emb = self.extractor(mel.transpose([0, 2, 1]))
        T_pred = content_emb.shape[1]
        T_target = hubert_emb.shape[1]
        if T_pred > T_target:
            content_emb = content_emb[:, :T_target, :]
        elif T_pred < T_target:
            content_emb = F.pad(content_emb, [0, 0, 0, T_target - T_pred])
        valid_mask = batch["valid_mask"].unsqueeze(-1)
        squared_error = ((content_emb - hubert_emb) ** 2) * valid_mask
        loss = squared_error.sum() / paddle.clip(valid_mask.sum() * content_emb.shape[-1], min=1.0)
        B, T, D = content_emb.shape
        valid_flat = valid_mask.squeeze(-1).reshape([-1]) > 0
        sim = F.cosine_similarity(
            content_emb.reshape([-1, D])[valid_flat],
            hubert_emb.reshape([-1, D])[valid_flat],
            axis=-1,
        ).mean()

        self.log("val/loss", loss, on_epoch=True, prog_bar=False, logger=False)
        self.log("val/cosine_sim", sim, on_epoch=True, prog_bar=False, logger=False)

    def on_validation_epoch_end(self) -> None:
        """Write epoch-mean val metrics to VisualDL."""
        if self.skip_immediate_validation:
            self.skip_immediate_validation = False
            return
        if self.logger is not None:
            step = self.global_step
            val_metrics = {}
            if self._trainer is not None:
                logged = self._trainer.callback_metrics
                for key in ("val/loss", "val/cosine_sim"):
                    if key in logged:
                        val_metrics[key] = logged[key]
                if not val_metrics and self._trainer._results is not None:
                    m = self._trainer._results.metrics(on_step=False)
                    for key in ("val/loss", "val/cosine_sim"):
                        if key in m["callback"]:
                            val_metrics[key] = m["callback"][key]
            if val_metrics:
                self.logger.log_metrics(val_metrics, step=step)

    def train_dataloader(self):
        """Training dataloader built from the config's ``data`` section."""
        data_cfg = self.config.get("data", {})
        audio_cfg = self.config.get("audio", {})
        dataset = ContentExtractorDataset(
            hdf5_path=data_cfg.get("hdf5_path", "data/libritts/hubert_embeddings/train.h5"),
            max_frames=audio_cfg.get("max_frames", 500),
        )
        return build_train_dataloader(dataset, self.config)

    def val_dataloader(self):
        """Validation dataloader — fixed sample count, batch size 1."""
        data_cfg = self.config.get("data", {})
        audio_cfg = self.config.get("audio", {})
        dataset = ContentExtractorDataset(
            hdf5_path=data_cfg.get("hdf5_path", "data/libritts/hubert_embeddings/train.h5"),
            max_frames=audio_cfg.get("max_frames", 500),
            max_samples=data_cfg.get("val_max_samples", 50),
        )
        return build_val_dataloader(dataset)

    def configure_optimizers(self):
        """Adam optimizer + StepDecay LR scheduler."""
        train_cfg = self.config.get("training", {})
        lr = float(train_cfg.get("learning_rate", 2e-4))
        step_size = int(train_cfg.get("step_size", 5000))
        gamma = float(train_cfg.get("gamma", 0.8))

        scheduler = paddle.optimizer.lr.StepDecay(
            learning_rate=lr,
            step_size=step_size,
            gamma=gamma,
        )

        opt = paddle.optimizer.Adam(
            learning_rate=scheduler,
            parameters=self.parameters(),
            beta1=0.9,
            beta2=0.98,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

