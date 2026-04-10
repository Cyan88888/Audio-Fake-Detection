"""Load HuBERTTransformerDetector from Lightning checkpoint or exported .pt bundle."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_detector_from_lightning_ckpt(
    ckpt_path: str,
    config_path: Optional[str] = None,
    map_location: Optional[str] = None,
) -> torch.nn.Module:
    """
    Loads weights under prefix ``detect_model.`` from a PL checkpoint.
    Uses ``config.yaml`` next to the experiment (same folder as checkpoints parent) to rebuild the module.
    """
    ckpt_path = Path(ckpt_path)
    if config_path is None:
        config_path = str(ckpt_path.parent.parent / "config.yaml")
    cfg = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(cfg.detect_model)

    ckpt = torch.load(str(ckpt_path), map_location=map_location or "cpu")
    state = ckpt.get("state_dict", ckpt)
    prefix = "detect_model."
    sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not sub:
        raise KeyError(f"No '{prefix}*' keys in checkpoint {ckpt_path}")
    model.load_state_dict(sub, strict=True)
    return model


def load_detector_bundle(
    pt_path: str,
    map_location: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    from safeear.models.detector_transformer import HuBERTTransformerDetector

    bundle = torch.load(pt_path, map_location=map_location or "cpu")
    if not isinstance(bundle, dict) or "state_dict" not in bundle:
        raise ValueError(f"Expected dict with 'state_dict' and 'arch' keys, got {type(bundle)}")
    arch = bundle.get("arch", {})
    model = HuBERTTransformerDetector(**arch)
    model.load_state_dict(bundle["state_dict"], strict=True)
    return model, bundle.get("meta", {})


def load_detector_auto(
    ckpt_or_pt: str,
    map_location: Optional[str] = None,
) -> torch.nn.Module:
    path = Path(ckpt_or_pt)
    if path.suffix == ".ckpt":
        return load_detector_from_lightning_ckpt(str(path), map_location=map_location)
    return load_detector_bundle(str(path), map_location=map_location)[0]
