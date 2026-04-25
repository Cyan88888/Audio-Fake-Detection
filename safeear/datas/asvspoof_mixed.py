from __future__ import annotations

from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .asvspoof19 import ASVSppof2019, collate_fn
from .asvspoof21 import ASVSppof2021


class DataClass:
    """Mixed-domain dataclass for quick joint training.

    Train = ASVspoof2019 train + ASVspoof2021 eval (as extra domain samples).
    Val/Test = ASVspoof2019 dev/eval for stable in-domain comparison.
    """

    def __init__(
        self,
        train19_path,
        train21_path,
        val19_path,
        test19_path,
        max_len: int = 64600,
        eval_return_full: bool = False,
        eval_crop_mode: str = "head",
        use_train21: bool = True,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.eval_return_full = bool(eval_return_full)
        self.eval_crop_mode = str(eval_crop_mode).lower()
        self.use_train21 = bool(use_train21)

        train_19 = ASVSppof2019(
            train19_path[0],
            train19_path[1],
            train19_path[2],
            max_len=self.max_len,
            is_train=True,
            eval_return_full=False,
            eval_crop_mode=self.eval_crop_mode,
        )
        if self.use_train21:
            train_21 = ASVSppof2021(
                train21_path[0],
                train21_path[1],
                train21_path[2],
                max_len=self.max_len,
                is_train=True,
                codec=False,
                eval_return_full=False,
            )
            self.train = ConcatDataset([train_19, train_21])
        else:
            self.train = train_19

        self.val = ASVSppof2019(
            val19_path[0],
            val19_path[1],
            val19_path[2],
            max_len=self.max_len,
            is_train=False,
            eval_return_full=self.eval_return_full,
            eval_crop_mode=self.eval_crop_mode,
        )
        self.test = ASVSppof2019(
            test19_path[0],
            test19_path[1],
            test19_path[2],
            max_len=self.max_len,
            is_train=False,
            eval_return_full=self.eval_return_full,
            eval_crop_mode=self.eval_crop_mode,
        )

    def __call__(self, mode: str):
        if mode == "train":
            return self.train
        if mode == "val":
            return self.val
        if mode == "test":
            return self.test
        raise ValueError(f"Unknown mode: {mode}.")


class DataModule(LightningDataModule):
    def __init__(self, DataClass_dict, batch_size, num_workers, pin_memory):
        super().__init__()
        self.save_hyperparameters(logger=False)
        DataClass_dict.pop("_target_")
        self.dataset_select = DataClass(**DataClass_dict)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage=None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.dataset_select("train")
            self.data_val = self.dataset_select("val")
            self.data_test = self.dataset_select("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )
