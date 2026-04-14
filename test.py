import importlib
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import argparse
import pytorch_lightning as pl
import torch
import hydra

torch.set_float32_matmul_precision("high")

from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

@rank_zero_only
def print_only(message: str):
    """Prints a message only on rank 0."""
    print(message)
    
def train(cfg: DictConfig, args) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    # instantiate datamodule
    print_only(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()

    print_only(f"Instantiating detect model <{cfg.detect_model._target_}>")
    detect_model: torch.nn.Module = hydra.utils.instantiate(cfg.detect_model)

    print_only(f"Instantiating system <{cfg.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(
        cfg.system,
        detect_model=detect_model,
    )
    
    # instantiate logger so validate/test metrics are persisted
    print_only(f"Instantiating logger <{cfg.logger._target_}>")
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "logs"), exist_ok=True)
    logger = hydra.utils.instantiate(cfg.logger)

    # instantiate trainer
    print_only(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    # Run validation first so dev-set threshold (e.g. val_mindcf) is stored for test acc_selected / f1_selected / etc.
    if args.ckpt_path:
        val_out = trainer.validate(system, datamodule=datamodule, ckpt_path=args.ckpt_path)
    else:
        val_out = trainer.validate(system, datamodule=datamodule)

    test_out = trainer.test(system, datamodule=datamodule, ckpt_path=args.ckpt_path)

    # Persist final metrics for easy experiment comparison
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    result_json = os.path.join(cfg.exp.dir, cfg.exp.name, "test_results.json")
    payload = {
        "ckpt_path": args.ckpt_path,
        "validate": val_out,
        "test": test_out,
    }
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    result_csv = os.path.join(cfg.exp.dir, cfg.exp.name, "test_results.csv")
    test_dict = test_out[0] if isinstance(test_out, list) and test_out else {}
    with open(result_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_dict.keys()))
        writer.writeheader()
        if test_dict:
            writer.writerow(test_dict)
    print_only(f"Saved test metrics to: {result_json} and {result_csv}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="local/conf.yml",
        help="Full path to save best validation model",
    )
    parser.add_argument(
        "--ckpt_path",
        help="Full path to save best validation model",
    )
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)
    
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    # 保存配置到新的文件
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))
    
    train(cfg, args)
    