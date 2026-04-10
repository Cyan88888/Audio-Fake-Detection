# <font color=E7595C>Safe</font><font color=F6C446>Ear</font><img src="assert/SafeEar_logo.jpg" alt="icon" style="width: 2em; height: 1.5em; vertical-align: middle;">: <font color=E7595C>Content Privacy-Preserving</font> <font color=F6C446>Audio Deepfake Detection</font>

[![arXiv](https://img.shields.io/badge/arXiv-2409.09272-b31b1b.svg)](https://arxiv.org/abs/2409.09272)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com) 
[![CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
![GitHub stars](https://img.shields.io/github/stars/LetterLiGo/SafeEar)
![GitHub forks](https://img.shields.io/github/forks/LetterLiGo/SafeEar)
![Website](https://img.shields.io/website?url=https://safeearweb.github.io/Project/)


By [1] Zhejiang University, [2] Tsinghua University.
* [Xinfeng Li](https://letterligo.github.io)* [1], [Kai Li](https://cslikai.cn)* [2], Yifan Zheng [1], Chen Yan† [1], Xiaoyu Ji [1], Wenyuan Xu [1].

This repository is an official implementation of the SafeEar accepted to **ACM CCS 2024** (Core-A*, CCF-A, Big4) .

Please also visit our <a href="https://safeearweb.github.io/Project/">(1) Project Website</a>, <a href="https://zenodo.org/records/14062964">(2) Full CVoiceFake Dataset</a>, and <a href="https://zenodo.org/records/11124319">(3) Sampled CVoiceFake Dataset</a>.

## 🔥News

[2025-03-18]: Supported the batch testing for ASVspoof 2019 and 2021, fixed some bugs for datasets and trainer.

[2024-12-10]: Fixed all the bugs for training and test, and uploaded the files for data generation `datas/`.

[2024-12-01]: Uploaded the checkpoint for data generation `datas/`.

## ✨Key Highlights:

In this paper, we propose SafeEar, a novel framework that aims to detect deepfake audios without relying on accessing the speech content within. Our key idea is to devise a neural audio codec into a novel decoupling model that well separates the semantic and acoustic information from audio samples, and only use the acoustic information (e.g., prosody and timbre) for deepfake detection. In this way, no semantic content will be exposed to the detector. To overcome the challenge of identifying diverse deepfake audio without semantic clues, we enhance our deepfake detector with multi-head self-attention and codec augmentation. Extensive experiments conducted on four benchmark datasets demonstrate SafeEar’s effectiveness in detecting various deepfake techniques with an equal error rate (EER) down to 2.02%. Simultaneously, it shields five-language speech content from being deciphered by both machine and human auditory analysis, demonstrated by word error rates (WERs) all above 93.93% and our user study. Furthermore, our benchmark constructed for anti-deepfake and anti-content recovery evaluation helps provide a basis for future research in the realms of audio privacy preservation and deepfake detection.

## 🚀Overall Pipeline

![pipeline](assert/overall.gif)

## 🔧Installation

1. Clone the repository:

```shell
git clone git@github.com:LetterLiGo/SafeEar.git
cd SafeEar/
```

2. Create and activate the conda environment:

```shell
conda create -n safeear python=3.9 
conda activate safeear
```

3. Install PyTorch and torchvision following the [official instructions](https://pytorch.org). The code requires `python=3.9`, `pytorch=1.13`, `torchvision=0.14`.


```shell
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

```
4. Install other dependencies:

```shell 
pip install pip==24.0
pip install -r requirements.txt
```

## 📊Model Performance
### ASVspoof 2019 & 2021
![](assert/ASVSpoof-results.png)
### Speech Recognition Performance
![](assert/exp1.png)

## Data preparation

### AVSpoof 2019 & 2021

Please download the [ASVspoof 2019](https://datashare.is.ed.ac.uk/handle/10283/3336) and [ASVspoof 2021](https://www.asvspoof.org/index2021.html) datasets and extract them to the `datas/datasets` directory.

```shell
datas/datasets/ASVspoof2019
datas/datasets/ASVspoof2021
```

#### Generate the Hubert L9 feature files

```shell
mkdir model_zoos
cd model_zoos
wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
wget https://cloud.tsinghua.edu.cn/f/413a0cd2e6f749eea956/?dl=1 -O SpeechTokenizer.pt
cd ../datas
# Generate the Hubert L9 feature files for ASVspoof 2019
python dump_hubert_avg_feature.py datasets/ASVSpoof2019 datasets/ASVSpoof2019_Hubert_L9
# Generate the Hubert L9 feature files for ASVspoof 2021
python dump_hubert_avg_feature.py datasets/ASVSpoof2021 datasets/ASVSpoof2021_Hubert_L9
```

## 📚Training

### Transformer-only spoof detection (no SpeechTokenizer / privacy pipeline)

This fork adds **frame-level features → Transformer classifier** paths suitable for thesis work focused on *Transformer + spoof detection* (no `SpeechTokenizer` decoupling).

- HuBERT features: [`config/transformer_spoof19.yaml`](config/transformer_spoof19.yaml)
- WavLM features (recommended for new runs): [`config/transformer_spoof19_wavlm.yaml`](config/transformer_spoof19_wavlm.yaml)
- Lightning module: `safeear.trainer.transformer_trainer.TransformerSpoofTrainer`
- Model: `safeear.models.detector_transformer.FrameTransformerDetector` (alias `HuBERTTransformerDetector`)

#### WavLM feature dump (train/eval `.npy` layout matches HuBERT pipeline)

From repo root, mirror each `.../flac` tree into `.../ASVSpoof2019_WavLM_base/...` (create parent dirs first):

```shell
pip install 'transformers>=4.30.0'
python datas/dump_wavlm_feature.py \
  datas/datasets/ASVSpoof2019/LA/ASVspoof2019_LA_train/flac \
  datas/datasets/ASVSpoof2019_WavLM_base/LA/ASVspoof2019_LA_train/flac
# Repeat for LA_dev and LA_eval flac roots, matching paths in transformer_spoof19_wavlm.yaml
```

Then:

```shell
python train.py --conf_dir config/transformer_spoof19_wavlm.yaml
python test.py --conf_dir config/transformer_spoof19_wavlm.yaml --ckpt_path Exps/TransformerSpoof19_wavlm_e30/checkpoints/<best>.ckpt
```

#### HuBERT L9 features (legacy / comparison)

Prepare HuBERT L9 `.npy` features as in **Data preparation**, then:

```shell
python train.py --conf_dir config/transformer_spoof19.yaml
python test.py --conf_dir config/transformer_spoof19.yaml --ckpt_path Exps/TransformerSpoof19_hubert_e30/checkpoints/<best>.ckpt
```

Set `SAFEAR_ASVSPOOF2019_ROOT` if your ASVspoof 2019 FLAC root is not `datas/datasets/ASVSpoof2019`.

### Original SafeEar paper configs (legacy)

The upstream README referred to `config/train19.yaml` / `config/train21.yaml`. If you still use **SpeechTokenizer + SafeEar1s** YAMLs under `Exps/`, you must restore the old `train.py`/`test.py` decouple block and `safeear.trainer.safeear_trainer` from the official repo; the default entrypoints above follow the **Transformer-only** layout.

## 📈Testing/Inference

### CLI prediction (WavLM or HuBERT on-the-fly + checkpoint)

Default uses WavLM (`microsoft/wavlm-base`, downloads on first run). For HuBERT, add `--feat hubert` and ensure `model_zoos/hubert_base_ls960.pt` exists.

```shell
python -m inference.predict --audio path/to/sample.wav --ckpt Exps/TransformerSpoof19_wavlm_e30/checkpoints/last.ckpt
python -m inference.predict --audio path/to/sample.wav --ckpt path.ckpt --feat hubert
```

### Export detector weights for deployment

```shell
python inference/export_weights.py --ckpt Exps/TransformerSpoof19_hubert_e30/checkpoints/last.ckpt --out inference/exports/detector.pt
```

### Web UI (upload audio)

```shell
export SAFEAR_CKPT=Exps/TransformerSpoof19_wavlm_e30/checkpoints/last.ckpt
# Optional: SAFEAR_FEAT=wavlm (default) or hubert
uvicorn web.api:app --host 0.0.0.0 --port 8080
```

Open `http://127.0.0.1:8080/` in a browser.

### Batch evaluation (Lightning)

```shell
python test.py --conf_dir config/transformer_spoof19.yaml --ckpt_path path/to.ckpt
```

## Bugs and Issues

If you meet `RuntimeError: Failed to load audio from <_io.BytesIO object at 0x7f45cb978f90>`, please use the following command to fix it:

```shell
conda install -c anaconda 'ffmpeg<4.4'
```

## 📜Citation

If you find our work/code/dataset helpful, please consider citing:

```
@inproceedings{li2024safeear,
  author       = {Li, Xinfeng and Li, Kai and Zheng, Yifan and Yan, Chen and Ji, Xiaoyu, and Xu, Wenyuan},
  title        = {{SafeEar: Content Privacy-Preserving Audio Deepfake Detection}},
  booktitle    = {Proceedings of the 2024 {ACM} {SIGSAC} Conference on Computer and Communications Security (CCS)}
  year         = {2024},
}
```
