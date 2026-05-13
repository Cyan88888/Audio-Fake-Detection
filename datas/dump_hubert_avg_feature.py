# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import tqdm
import fairseq
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().to(self.device)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")
        logger.info(f" device = {self.device}")

    def read_audio(self, path, ref_len=None):
        wav, sr = librosa.load(path, sr=None)
        assert sr == self.task.cfg.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(self.device)
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            avg_feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _, avg_feat_chunk = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                avg_feat.append(avg_feat_chunk)
        return torch.cat(avg_feat, 1).squeeze(0)

def dump_feature(reader, audio_dir, save_dir, skip_exists=False):
    save_dir = Path(save_dir)
    audio_dir = Path(audio_dir)
    
    audio_files = list(audio_dir.glob("**/*.flac"))
    for audio_file in tqdm.tqdm(audio_files):
        releative_path = audio_file.relative_to(audio_dir).with_suffix(".npy")
        save_path = save_dir / releative_path
        if skip_exists and save_path.exists():
            continue
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        
        feat = reader.get_feats(audio_file)
        np.save(save_path, feat.cpu().numpy().astype("float32"))
    logger.info("finished successfully")

def main(audio_dir, save_dir, ckpt_path, layer, max_chunk, device=None, skip_exists=False):
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk, device=device)
    dump_feature(reader, audio_dir, save_dir, skip_exists=skip_exists)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", nargs="?", default="datasets/ASVSpoof2019", help="Directory containing audio files")
    parser.add_argument("save_dir", nargs="?", default="datasets/ASVSpoof2019_Hubert_L9", help="Directory to save extracted features")
    parser.add_argument("ckpt_path", nargs="?", default="../model_zoos/hubert_base_ls960.pt", help="Path to the checkpoint file")
    parser.add_argument("layer", nargs="?", type=int, default=9, help="Layer number to extract features from")
    parser.add_argument("--max_chunk", type=int, default=1600000, help="Maximum chunk size for processing")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, or cpu. Defaults to CUDA if available.")
    parser.add_argument("--skip_exists", action="store_true", help="Skip feature files that already exist.")
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
