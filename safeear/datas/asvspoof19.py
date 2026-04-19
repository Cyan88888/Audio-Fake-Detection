import glob
import random
import os
from pathlib import Path
from typing import Optional
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchaudio.functional

_REPO_ROOT = Path(__file__).resolve().parents[2]

def get_path_iterator(tsv):
    """
    读取TSV文件，跳过第一行硬编码路径，返回空root和文件列表
    """
    with open(tsv, "r") as f:
        f.readline()  # 跳过第一行（作者的绝对路径）
        lines = [line.strip() for line in f]
    root = ""  # 清空root，后续用配置的路径拼接
    return root, lines

def load_feature(feat_path):
    """加载Hubert特征"""
    feat = np.load(feat_path, mmap_mode="r")
    return feat

class ASVSppof2019(Dataset):
    def __init__(
        self,
        tsv_path,
        protocol_path,
        feat_dir,
        max_len=64600,
        is_train=True,
        eval_return_full=False,
        online_train_feature_extraction=False,
        wavlm_model_name="microsoft/wavlm-base",
        online_codec_aug_prob=0.0,
        online_codec_formats=None,
    ):
        super().__init__()
        # 读取TSV文件（仅获取文件名，不使用TSV中的root）
        _, self.lines = get_path_iterator(tsv_path)
        
        # 关键修正：分别定义音频根路径和特征根路径
        self.feat_dir = Path(feat_dir)  # 配置文件传入的Hubert特征路径
        self.audio_root = Path(
            os.environ.get("SAFEAR_ASVSPOOF2019_ROOT", _REPO_ROOT / "datas" / "datasets" / "ASVSpoof2019")
        )
        self.max_len = max_len 
        self.is_train = is_train
        self.eval_return_full = bool(eval_return_full)
        self.online_train_feature_extraction = bool(online_train_feature_extraction)
        self.wavlm_model_name = wavlm_model_name
        self.online_codec_aug_prob = float(online_codec_aug_prob)
        if online_codec_formats is None:
            online_codec_formats = ["ogg"]
        self.online_codec_formats = list(online_codec_formats)
        self._online_codec_formats_active = list(self.online_codec_formats)
        self._online_featurizer: Optional[object] = None
        
        # 核心修复：根据数据集类型选择正确的音频子目录
        if "train" in str(feat_dir) or "train" in str(tsv_path):
            self.audio_subdir = "LA/ASVspoof2019_LA_train/flac"
            self.dataset_type = "train"
        elif "dev" in str(feat_dir) or "dev" in str(tsv_path):
            self.audio_subdir = "LA/ASVspoof2019_LA_dev/flac"
            self.dataset_type = "val"
        elif "eval" in str(feat_dir) or "eval" in str(tsv_path):
            self.audio_subdir = "LA/ASVspoof2019_LA_eval/flac"
            self.dataset_type = "test"
        else:
            raise ValueError(f"无法识别数据集类型！特征路径：{feat_dir} | TSV路径：{tsv_path}")
        
        # 加载第一个音频文件获取采样率（使用正确的子目录）
        first_audio_name = self.lines[0].split('\t')[0]
        first_audio_path = self.audio_root / self.audio_subdir / first_audio_name
        
        # 调试打印
        print(f"[DEBUG] 数据集类型：{self.dataset_type}")
        print(f"[DEBUG] 音频子目录：{self.audio_subdir}")
        print(f"[DEBUG] 第一个音频路径：{first_audio_path}")
        
        # 加载音频（增加异常提示）
        try:
            _, self.sr = torchaudio.load(str(first_audio_path))
        except Exception as e:
            raise RuntimeError(f"加载采样率失败！请检查路径是否正确：{first_audio_path}\n错误详情：{e}")
        
        # 加载标签文件
        with open(protocol_path) as file:
            meta_infos = file.readlines()
        self.meta_infos = meta_infos
        self.mapping = {
            meta_info.replace('\n', '').split(' ')[1]: meta_info.replace('\n', '').split(' ')[-1]
            for meta_info in meta_infos
        }

    def _get_online_featurizer(self):
        if self._online_featurizer is None:
            # Keep online extraction on CPU to avoid CUDA in dataloader workers.
            from inference.wavlm_featurizer import WavLMFeaturizer

            self._online_featurizer = WavLMFeaturizer(
                model_name=self.wavlm_model_name,
                device=torch.device("cpu"),
            )
        return self._online_featurizer

    def _maybe_apply_codec(self, audio: torch.Tensor) -> torch.Tensor:
        if self.online_codec_aug_prob <= 0.0:
            return audio
        if random.random() >= self.online_codec_aug_prob:
            return audio
        if not self._online_codec_formats_active:
            return audio

        codec = random.choice(self._online_codec_formats_active)
        try:
            if codec == "gsm":
                x = torchaudio.functional.resample(audio, self.sr, 8000)
                x = torchaudio.functional.apply_codec(x, 8000, codec)
                return torchaudio.functional.resample(x, 8000, self.sr)
            return torchaudio.functional.apply_codec(audio, self.sr, codec)
        except Exception:
            # Disable unsupported codec after first failure to avoid repeated stderr spam.
            self._online_codec_formats_active = [
                c for c in self._online_codec_formats_active if c != codec
            ]
            return audio

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        feat_duration = self.max_len // 320
    
        # 获取文件名（不含路径）
        audio_filename = self.lines[index].split('\t')[0]
        # 拼接特征路径（Hubert特征是.npy文件）
        feat_path = self.feat_dir / audio_filename.replace(".flac", ".npy")
        # 拼接音频路径（使用初始化时确定的子目录）
        audio_path = self.audio_root / self.audio_subdir / audio_filename
        
        # 加载音频（增加异常处理）
        try:
            audio = torchaudio.load(str(audio_path))[0]
        except Exception as e:
            raise RuntimeError(f"加载文件失败！\n音频路径：{audio_path}\n错误：{e}")
        
        # 获取标签
        waveform_info = self.mapping[audio_filename.split('.')[0]]
        target = 1 if waveform_info == 'spoof' else 0 

        # 在线训练特征提取：训练集使用音频增强 + WavLM实时提取
        if self.is_train and self.online_train_feature_extraction:
            audio_aug = self._maybe_apply_codec(audio)
            if audio_aug.shape[1] > self.max_len:
                st = random.randint(0, audio_aug.shape[1] - self.max_len - 1)
                audio_aug = audio_aug[:, st : st + self.max_len]
            elif audio_aug.shape[1] < self.max_len:
                audio_aug = torch.nn.functional.pad(
                    audio_aug, (0, self.max_len - audio_aug.shape[1]), "constant", 0
                )

            featurizer = self._get_online_featurizer()
            feat = featurizer.wav_tensor_to_feat(
                audio_aug.squeeze(0),
                max_len=self.max_len,
                preserve_length=False,
            )
            avg_hubert_feat = feat.squeeze(0).to(torch.float32)  # [C, T]
            return audio_aug, avg_hubert_feat, target

        # 离线特征路径（验证/测试，及默认训练流程）
        try:
            avg_hubert_feat = torch.tensor(load_feature(str(feat_path)))
        except Exception as e:
            raise RuntimeError(f"加载特征失败！\n特征路径：{feat_path}\n错误：{e}")
        
        # 调整特征维度
        if avg_hubert_feat.ndim == 3:
            avg_hubert_feat = avg_hubert_feat.permute(2, 1, 0).squeeze(1)
        else:
            avg_hubert_feat = avg_hubert_feat.permute(1, 0)
        
        # 训练集：随机裁剪
        if self.is_train and audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1] - self.max_len - 1)
            feat_st = st // 320
            ed = st + self.max_len
            # 裁剪/填充特征
            if avg_hubert_feat[:, feat_st:feat_st + feat_duration].shape[1] < feat_duration:
                avg_hubert_feat = avg_hubert_feat[:, feat_st:feat_st + feat_duration]
                avg_hubert_feat = torch.nn.functional.pad(avg_hubert_feat, (0, feat_duration - avg_hubert_feat.shape[1]), "constant", 0)
                return audio[:, st:ed], avg_hubert_feat, target
            else:
                return audio[:, st:ed], avg_hubert_feat[:, feat_st:feat_st + feat_duration], target
        
        # 验证/测试集：若开启 eval_return_full，直接返回完整特征给 TTA 切段
        if not self.is_train and self.eval_return_full:
            return audio, avg_hubert_feat, target, str(audio_path)

        # 验证/测试集：固定裁剪
        if not self.is_train and audio.shape[1] > self.max_len:
            st = 0
            feat_st = 0
            ed = st + self.max_len
            if avg_hubert_feat[:, feat_st:feat_st + feat_duration].shape[1] < feat_duration:
                avg_hubert_feat = avg_hubert_feat[:, feat_st:feat_st + feat_duration]
                avg_hubert_feat = torch.nn.functional.pad(avg_hubert_feat, (0, feat_duration - avg_hubert_feat.shape[1]), "constant", 0)
                return audio[:, st:ed], avg_hubert_feat, target, str(audio_path)
            else:
                return audio[:, st:ed], avg_hubert_feat[:, feat_st:feat_st + feat_duration], target, str(audio_path)

        # 填充短音频
        if audio.shape[1] < self.max_len:
            audio_pad_length = self.max_len - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, audio_pad_length), "constant", 0)
        
        # 填充短特征
        if avg_hubert_feat.shape[1] < feat_duration:
            avg_hubert_feat = torch.nn.functional.pad(avg_hubert_feat, (0, feat_duration - avg_hubert_feat.shape[1]), "constant", 0)

        if not self.is_train:
            return audio, avg_hubert_feat, target, str(audio_path)

        return audio, avg_hubert_feat, target
    
def pad_sequence(batch):
    """填充序列到相同长度"""
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)

def collate_fn(batch):
    """批处理函数"""
    wavs = []
    feats = []
    targets = []
    audio_paths = []
    feat_lengths = []
    for item in batch:
        # 兼容测试集返回值（多了audio_path）
        if len(item) == 4:
            wav, feat, target, audio_path = item
            audio_paths.append(audio_path)
        else:
            wav, feat, target = item
        wavs.append(wav)
        feats.append(feat)
        targets.append(target)
        feat_lengths.append(feat.shape[1])

    wavs = pad_sequence(wavs)
    feats = pad_sequence(feats)
    target_tensor = torch.tensor(targets).long()
    feat_len_tensor = torch.tensor(feat_lengths).long()
    if audio_paths:
        return wavs, feats, target_tensor, audio_paths, feat_len_tensor
    return wavs, feats, target_tensor

class DataClass:
    def __init__(
        self,
        train_path, 
        val_path, 
        test_path, 
        max_len=64600,
        eval_return_full=False,
        online_train_feature_extraction=False,
        wavlm_model_name="microsoft/wavlm-base",
        online_codec_aug_prob=0.0,
        online_codec_formats=None,
    ) -> None:
        super().__init__()

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.max_len = max_len
        self.eval_return_full = bool(eval_return_full)
        self.online_train_feature_extraction = bool(online_train_feature_extraction)
        self.wavlm_model_name = wavlm_model_name
        self.online_codec_aug_prob = float(online_codec_aug_prob)
        self.online_codec_formats = online_codec_formats

        # 初始化数据集（传入正确的路径）
        self.train = ASVSppof2019(
            self.train_path[0], 
            self.train_path[1], 
            self.train_path[2], 
            self.max_len, 
            is_train=True,
            online_train_feature_extraction=self.online_train_feature_extraction,
            wavlm_model_name=self.wavlm_model_name,
            online_codec_aug_prob=self.online_codec_aug_prob,
            online_codec_formats=self.online_codec_formats,
        )
        self.val = ASVSppof2019(
            self.val_path[0], 
            self.val_path[1], 
            self.val_path[2], 
            self.max_len, 
            is_train=False,  # 验证集设为False
            eval_return_full=self.eval_return_full,
        )
        self.test = ASVSppof2019(
            self.test_path[0], 
            self.test_path[1], 
            self.test_path[2],
            self.max_len,
            is_train=False,
            eval_return_full=self.eval_return_full,
        )
    
    def __call__(self, mode: str) -> ASVSppof2019:
        if mode == "train":
            return self.train
        elif mode == "val":
            return self.val
        elif mode == "test":
            return self.test
        else:
            raise ValueError(f"Unknown mode: {mode}.")

class DataModule(LightningDataModule):
    def __init__(self, DataClass_dict, batch_size, num_workers, pin_memory):
        super().__init__()
        self.save_hyperparameters(logger=False)
        DataClass_dict.pop("_target_")
        self.dataset_select = DataClass(**DataClass_dict)

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self, stage = None) -> None:
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
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn  # 验证集也用collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn  # 测试集也用collate_fn
        )