"""
Lightning module: HuBERT (or frame) features -> Transformer spoof detector.
No SpeechTokenizer / privacy decoupling.
"""
from __future__ import annotations

import math
import os
from typing import Any, List, Optional

import json

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.trainer.states import TrainerFn

from ..losses.loss import compute_eer
from ..utils.binary_metrics import compute_binary_classification_metrics, compute_min_dcf
from ..utils.tdcf_metrics import compute_min_tdcf_from_cm_bonafide_score, load_asvspoof2019_la_asv_scores


def _get_feat_target_batch(batch):
    """Returns (feat, target, audio_path_or_none, feat_lengths_or_none)."""
    if len(batch) == 5:
        _, feat, target, audio_path, feat_lengths = batch
        return feat, target, audio_path, feat_lengths
    if len(batch) == 4:
        _, feat, target, audio_path = batch
        return feat, target, audio_path, None
    _, feat, target = batch
    return feat, target, None, None


def adjust_learning_rate(optimizer, epoch: int, lr: float, warmup: int, epochs: int):
    lr = lr
    if epoch < warmup:
        lr = lr / max(1, (warmup - epoch))
    else:
        lr *= 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup) / max(1, (epochs - warmup))))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _unwrap_optimizer(optimizers):
    """Lightning may return a single optimizer or a list/tuple."""
    if optimizers is None:
        return None
    if isinstance(optimizers, (list, tuple)):
        return optimizers[0] if optimizers else None
    return optimizers


class TransformerSpoofTrainer(pl.LightningModule):
    def __init__(
        self,
        detect_model: torch.nn.Module,
        lr: float = 3e-4,
        save_score_path: Optional[str] = None,
        threshold_mode: str = "val_mindcf",
        threshold_fixed_bonafide: float = 0.5,
        p_target: float = 0.05,
        use_min_tdcf: bool = True,
        tdcf_la_root: str = "datas/datasets/ASVSpoof2019/LA",
        tdcf_p_tar: float = 0.9405,
        tdcf_p_non: float = 0.0095,
        tdcf_p_spoof: float = 0.05,
        tdcf_c_miss_asv: float = 1.0,
        tdcf_c_fa_asv: float = 10.0,
        tdcf_c_miss_cm: float = 1.0,
        tdcf_c_fa_cm: float = 10.0,
        label_smoothing: float = 0.0,
        aug_time_mask_prob: float = 0.0,
        aug_time_mask_max_frames: int = 0,
        test_tta_num_segments: int = 1,
        test_tta_segment_frames: int = 0,
        warmup_epochs_ratio: float = 0.1,
        weight_decay: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["detect_model"])
        self.detect_model = detect_model
        self.lr = lr
        self.save_score_path = save_score_path
        self.threshold_mode = threshold_mode
        self.threshold_fixed_bonafide = threshold_fixed_bonafide
        self.p_target = p_target
        self.use_min_tdcf = use_min_tdcf
        self.tdcf_p_tar = tdcf_p_tar
        self.tdcf_p_non = tdcf_p_non
        self.tdcf_p_spoof = tdcf_p_spoof
        self.tdcf_c_miss_asv = tdcf_c_miss_asv
        self.tdcf_c_fa_asv = tdcf_c_fa_asv
        self.tdcf_c_miss_cm = tdcf_c_miss_cm
        self.tdcf_c_fa_cm = tdcf_c_fa_cm
        self.label_smoothing = float(label_smoothing)
        self.aug_time_mask_prob = float(aug_time_mask_prob)
        self.aug_time_mask_max_frames = int(aug_time_mask_max_frames)
        self.test_tta_num_segments = int(test_tta_num_segments)
        self.test_tta_segment_frames = int(test_tta_segment_frames)
        self.warmup_epochs_ratio = warmup_epochs_ratio
        self.weight_decay = weight_decay
        self._asv_scores = None
        if self.use_min_tdcf:
            try:
                self._asv_scores = load_asvspoof2019_la_asv_scores(tdcf_la_root)
            except Exception as e:
                self.print(f"[WARN] Failed to load ASV scores for min-tDCF from {tdcf_la_root}: {e}")
                self._asv_scores = None

        self.val_index_loader: List[torch.Tensor] = []
        self.val_score_loader: List[torch.Tensor] = []
        self.val_loss_loader: List[torch.Tensor] = []
        self.eval_index_loader: List[torch.Tensor] = []
        self.eval_score_loader: List[torch.Tensor] = []
        self.eval_loss_loader: List[torch.Tensor] = []
        self.eval_filename_loader: List[Any] = []

    def _apply_time_mask(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply simple per-utterance time masking on (B, C, T) features."""
        if self.aug_time_mask_prob <= 0.0 or self.aug_time_mask_max_frames <= 0:
            return feat
        if feat.dim() != 3:
            return feat
        bsz, _, t = feat.shape
        if t <= 1:
            return feat
        max_w = min(self.aug_time_mask_max_frames, t)
        if max_w <= 0:
            return feat

        out = feat.clone()
        mask_flags = torch.rand(bsz, device=feat.device) < self.aug_time_mask_prob
        mask_ids = torch.nonzero(mask_flags, as_tuple=False).view(-1)
        for b in mask_ids.tolist():
            w = int(torch.randint(1, max_w + 1, (1,), device=feat.device).item())
            if w >= t:
                out[b] = 0.0
                continue
            s = int(torch.randint(0, t - w + 1, (1,), device=feat.device).item())
            out[b, :, s : s + w] = 0.0
        return out

    def forward(self, batch, is_train: bool = True):
        feat, target, audio_path, _ = _get_feat_target_batch(batch)
        feat = feat.to(memory_format=torch.contiguous_format).float()
        if is_train:
            feat = self._apply_time_mask(feat)
        target = target.long()
        logits, _ = self.detect_model(feat)
        if is_train:
            loss = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
            return loss, logits, target
        with torch.no_grad():
            prob_bonafide = torch.softmax(logits, dim=-1)[:, 0]
        return audio_path, torch.tensor(0.0, device=feat.device), prob_bonafide, target

    def _infer_tta_segment_frames(self) -> int:
        if self.test_tta_segment_frames > 0:
            return self.test_tta_segment_frames
        max_len = int(getattr(self.hparams, "max_len", 64600))
        return max(1, max_len // 320)

    def _tta_predict(self, feat: torch.Tensor, feat_lengths: Optional[torch.Tensor]) -> torch.Tensor:
        """TTA over multiple temporal segments; returns P(bonafide) per sample."""
        num_segments = max(1, self.test_tta_num_segments)
        seg_frames = self._infer_tta_segment_frames()
        probs = []
        bsz = feat.shape[0]
        for i in range(bsz):
            valid_t = int(feat.shape[-1])
            if feat_lengths is not None:
                valid_t = int(feat_lengths[i].item())
            valid_t = max(1, valid_t)
            cur = feat[i : i + 1, :, :valid_t]
            cur_t = cur.shape[-1]
            if cur_t <= seg_frames:
                segments = cur
            else:
                starts = torch.linspace(
                    0,
                    cur_t - seg_frames,
                    steps=num_segments,
                    device=feat.device,
                ).round().long().tolist()
                starts = list(dict.fromkeys(starts))
                seg_list = [cur[0, :, s : s + seg_frames] for s in starts]
                segments = torch.stack(seg_list, dim=0)
            logits_seg, _ = self.detect_model(segments)
            prob_seg = torch.softmax(logits_seg, dim=-1)[:, 0]
            probs.append(prob_seg.mean())
        return torch.stack(probs, dim=0)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self(batch, is_train=True)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        feat, target, _, _ = _get_feat_target_batch(batch)
        feat = feat.to(memory_format=torch.contiguous_format).float()
        target = target.long()
        with torch.no_grad():
            logits, _ = self.detect_model(feat)
            prob_bonafide = torch.softmax(logits, dim=-1)[:, 0]
            val_loss = F.cross_entropy(logits, target)
        self.val_index_loader.append(target)
        self.val_score_loader.append(prob_bonafide)
        self.val_loss_loader.append(val_loss.detach())

    def on_validation_epoch_end(self):
        if not self.val_index_loader:
            return
        all_index = self.all_gather(torch.cat(self.val_index_loader, dim=0)).view(-1).cpu().numpy()
        all_score = self.all_gather(torch.cat(self.val_score_loader, dim=0)).view(-1).cpu().numpy()

        # Use a fixed score direction for the primary metric to avoid optimistic
        # "oracle polarity" reporting. `all_score` is P(bonafide), and label 0 is bonafide.
        val_eer = compute_eer(all_score[all_index == 0], all_score[all_index == 1])[0]
        other_val_eer = compute_eer(-all_score[all_index == 0], -all_score[all_index == 1])[0]

        metrics_05 = compute_binary_classification_metrics(
            y_true=all_index,
            y_score_bonafide=all_score,
            threshold_bonafide=self.threshold_fixed_bonafide,
            pos_label=1,
        )
        min_dcf, min_dcf_th = compute_min_dcf(
            target_scores=all_score[all_index == 0],
            nontarget_scores=all_score[all_index == 1],
            p_target=self.p_target,
            c_miss=1.0,
            c_fa=1.0,
        )
        min_tdcf, min_tdcf_th = float("nan"), float("nan")
        if self._asv_scores is not None:
            min_tdcf, min_tdcf_th = compute_min_tdcf_from_cm_bonafide_score(
                cm_bonafide_scores=all_score,
                cm_labels=all_index,
                asv_target_scores=self._asv_scores["dev"]["target"],
                asv_nontarget_scores=self._asv_scores["dev"]["nontarget"],
                asv_spoof_scores=self._asv_scores["dev"]["spoof"],
                p_tar=self.tdcf_p_tar,
                p_non=self.tdcf_p_non,
                p_spoof=self.tdcf_p_spoof,
                c_miss_asv=self.tdcf_c_miss_asv,
                c_fa_asv=self.tdcf_c_fa_asv,
                c_miss_cm=self.tdcf_c_miss_cm,
                c_fa_cm=self.tdcf_c_fa_cm,
            )

        self.log_dict(
            {
                "val_loss": torch.stack(self.val_loss_loader).mean(),
                "val_eer": val_eer,
                "val_eer_reverse_score": other_val_eer,
                "val_acc": metrics_05["acc"],
                "val_precision": metrics_05["precision"],
                "val_recall": metrics_05["recall"],
                "val_f1": metrics_05["f1"],
                "val_roc_auc": metrics_05["roc_auc"],
                "val_pr_ap": metrics_05["pr_ap"],
                "val_minDCF": min_dcf,
                "val_minDCF_th": float(min_dcf_th),
                "val_min_tDCF": min_tdcf,
                "val_min_tDCF_th": min_tdcf_th,
            },
            sync_dist=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.val_index_loader.clear()
        self.val_score_loader.clear()
        self.val_loss_loader.clear()

        # LR logging / schedule only during fit(); validate()/test() may return optimizers as a list or have no training step.
        if getattr(self.trainer.state, "fn", None) != TrainerFn.FITTING:
            return
        opt = _unwrap_optimizer(self.optimizers())
        if opt is None:
            return
        self.log(
            "lr",
            opt.param_groups[0]["lr"],
            sync_dist=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        max_ep = self.trainer.max_epochs or 1
        warmup = int(max_ep * self.warmup_epochs_ratio)
        adjust_learning_rate(opt, self.current_epoch, self.lr, warmup, max_ep)

    def test_step(self, batch, batch_idx):
        feat, target, audio_path, feat_lengths = _get_feat_target_batch(batch)
        feat = feat.to(memory_format=torch.contiguous_format).float()
        target = target.long()
        if feat_lengths is not None:
            feat_lengths = feat_lengths.to(feat.device)
        with torch.no_grad():
            prob_bonafide = self._tta_predict(feat, feat_lengths)
            eps = 1e-6
            prob_bonafide = torch.clamp(prob_bonafide, eps, 1.0 - eps)
            logits_for_loss = torch.stack(
                [torch.log(prob_bonafide), torch.log(1.0 - prob_bonafide)],
                dim=-1,
            )
            test_loss = F.cross_entropy(logits_for_loss, target)
        self.eval_index_loader.append(target)
        self.eval_score_loader.append(prob_bonafide)
        self.eval_loss_loader.append(test_loss.detach())
        self.eval_filename_loader.append(audio_path)

    def on_test_epoch_end(self):
        if not self.eval_index_loader:
            return

        all_index = self.all_gather(torch.cat(self.eval_index_loader, dim=0)).view(-1).cpu().numpy()
        all_score = self.all_gather(torch.cat(self.eval_score_loader, dim=0)).view(-1).cpu().numpy()

        # Keep the same fixed direction as validation for comparability.
        eval_eer = compute_eer(all_score[all_index == 0], all_score[all_index == 1])[0]
        other_eval_eer = compute_eer(-all_score[all_index == 0], -all_score[all_index == 1])[0]

        metrics_05 = compute_binary_classification_metrics(
            y_true=all_index,
            y_score_bonafide=all_score,
            threshold_bonafide=self.threshold_fixed_bonafide,
            pos_label=1,
        )
        min_dcf, min_dcf_th = compute_min_dcf(
            target_scores=all_score[all_index == 0],
            nontarget_scores=all_score[all_index == 1],
            p_target=self.p_target,
            c_miss=1.0,
            c_fa=1.0,
        )
        min_tdcf, min_tdcf_th = float("nan"), float("nan")
        if self._asv_scores is not None:
            min_tdcf, min_tdcf_th = compute_min_tdcf_from_cm_bonafide_score(
                cm_bonafide_scores=all_score,
                cm_labels=all_index,
                asv_target_scores=self._asv_scores["eval"]["target"],
                asv_nontarget_scores=self._asv_scores["eval"]["nontarget"],
                asv_spoof_scores=self._asv_scores["eval"]["spoof"],
                p_tar=self.tdcf_p_tar,
                p_non=self.tdcf_p_non,
                p_spoof=self.tdcf_p_spoof,
                c_miss_asv=self.tdcf_c_miss_asv,
                c_fa_asv=self.tdcf_c_fa_asv,
                c_miss_cm=self.tdcf_c_miss_cm,
                c_fa_cm=self.tdcf_c_fa_cm,
            )

        if self.save_score_path:
            os.makedirs(self.save_score_path, exist_ok=True)
            fpr, tpr, roc_th = metrics_05["roc_curve"]
            pr_recall, pr_precision, pr_th = metrics_05["pr_curve"]
            np.save(os.path.join(self.save_score_path, "roc_curve.npy"), np.stack([fpr, tpr], axis=0))
            np.save(os.path.join(self.save_score_path, "roc_thresholds.npy"), roc_th)
            np.save(os.path.join(self.save_score_path, "pr_curve.npy"), np.stack([pr_recall, pr_precision], axis=0))
            np.save(os.path.join(self.save_score_path, "pr_thresholds.npy"), pr_th)
            # For offline plots (e.g. scripts/plot_experiment.py confusion matrix)
            np.save(os.path.join(self.save_score_path, "test_labels.npy"), all_index.astype(np.int64))
            np.save(os.path.join(self.save_score_path, "test_prob_bonafide.npy"), all_score.astype(np.float64))
            meta_path = os.path.join(self.save_score_path, "test_cm_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "threshold_bonafide": float(self.threshold_fixed_bonafide),
                        "test_tta_num_segments": int(self.test_tta_num_segments),
                        "test_tta_segment_frames": int(self._infer_tta_segment_frames()),
                    },
                    f,
                    indent=2,
                )

        self.log_dict(
            {
                "test_loss": torch.stack(self.eval_loss_loader).mean(),
                "test_eer": eval_eer,
                "test_eer_reverse_score": other_eval_eer,
                "test_acc": metrics_05["acc"],
                "test_precision": metrics_05["precision"],
                "test_recall": metrics_05["recall"],
                "test_f1": metrics_05["f1"],
                "test_roc_auc": metrics_05["roc_auc"],
                "test_pr_ap": metrics_05["pr_ap"],
                "test_minDCF": min_dcf,
                "test_minDCF_th": float(min_dcf_th),
                "test_min_tDCF": min_tdcf,
                "test_min_tDCF_th": min_tdcf_th,
            },
            sync_dist=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.eval_index_loader.clear()
        self.eval_score_loader.clear()
        self.eval_loss_loader.clear()
        self.eval_filename_loader.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.detect_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
