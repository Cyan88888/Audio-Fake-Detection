下面按 **「环境 → 数据与特征 → 训练 → 测试 → 可视化 →（可选）推理 / Web」** 给出完整流程：**每个代码块里只有一条命令**（或一条带续行的单条命令）。默认在 **Linux**、仓库根目录 **`SafeEar/`** 下执行；路径请按你机器上的实际位置修改。

---

**进入项目根目录**

```bash
cd /root/autodl-tmp/SafeEar
```

---

**（可选）创建并激活 Conda 环境**

```bash
conda create -n safeear python=3.9 -y
```

```bash
conda activate safeear
```

---

**安装 PyTorch（按你 CUDA 版本二选一；以下为 cu116 示例）**

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

---

**安装项目依赖**

```bash
pip install pip==24.0
```

```bash
pip install -r requirements.txt
```

---

**（可选）Hugging Face 下载慢时指定镜像**

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

**（若 ASVspoof2019 不在默认 `datas/datasets/ASVSpoof2019`）设置音频根目录**

```bash
export SAFEAR_ASVSPOOF2019_ROOT=/path/to/your/ASVSpoof2019
```

---

**准备协议与列表（需已下载 LA 数据；仓库内 `datas/ASVSpoof2019/` 等应已就绪）**

（无单独命令：请确认 `datas/datasets/ASVSpoof2019/LA/` 下存在 `ASVspoof2019_LA_train|dev|eval` 的 `flac`。）

---

**离线导出 WavLM 特征：训练集**

```bash
python datas/dump_wavlm_feature.py datas/datasets/ASVSpoof2019/LA/ASVspoof2019_LA_train/flac datas/datasets/ASVSpoof2019_WavLM_base/LA/ASVspoof2019_LA_train/flac
```

---

**离线导出 WavLM 特征：开发集**

```bash
python datas/dump_wavlm_feature.py datas/datasets/ASVSpoof2019/LA/ASVspoof2019_LA_dev/flac datas/datasets/ASVSpoof2019_WavLM_base/LA/ASVspoof2019_LA_dev/flac
```

---

**离线导出 WavLM 特征：评测集**

```bash
python datas/dump_wavlm_feature.py datas/datasets/ASVSpoof2019/LA/ASVspoof2019_LA_eval/flac datas/datasets/ASVSpoof2019_WavLM_base/LA/ASVspoof2019_LA_eval/flac
```

---

**训练（WavLM + Transformer，配置见 `config/transformer_spoof19_wavlm.yaml`）**

```bash
python train.py --conf_dir config/transformer_spoof19_wavlm.yaml
```

---

**测试（将 `ckpt` 换成你 `Exps/TransformerSpoof19_wavlm_e30/checkpoints/` 下实际最优文件）**

```bash
python test.py --conf_dir config/transformer_spoof19_wavlm.yaml --ckpt_path Exps/TransformerSpoof19_wavlm_e30/checkpoints/epoch=17-val_eer=0.0338.ckpt
```

---

**生成训练/测试曲线与混淆矩阵图（需已跑过测试以生成 `test_labels.npy` 等；见下说明）**

```bash
python scripts/plot_experiment.py --exp Exps/TransformerSpoof19_wavlm_e30
```

说明：混淆矩阵依赖 **`transformer_trainer` 在 `save_score_path` 非空时** 写出的 `test_labels.npy` / `test_prob_bonafide.npy`；你当前配置里 `save_score_path` 指向实验目录，**跑完上面 `test.py` 后**再执行本脚本即可。

---

**（可选）单条音频命令行推理**

```bash
python -m inference.predict --audio path/to/sample.wav --ckpt Exps/TransformerSpoof19_wavlm_e30/checkpoints/epoch=17-val_eer=0.0338.ckpt
```

---

**（可选）导出部署用权重**

```bash
python inference/export_weights.py --ckpt Exps/TransformerSpoof19_wavlm_e30/checkpoints/epoch=17-val_eer=0.0338.ckpt --out inference/exports/detector.pt
```

---

**（可选）启动 Web 推理**

```bash
export SAFEAR_CKPT=Exps/TransformerSpoof19_wavlm_e30/checkpoints/epoch=17-val_eer=0.0338.ckpt
```

```bash
uvicorn web.api:app --host 0.0.0.0 --port 8080
```

---

**说明（避免踩坑）**

1. **`--ckpt_path`**：必须指向你本机 `checkpoints/` 里真实存在的 `.ckpt`；若文件名不同，只改这一条里的路径即可。  
2. **`test.py`**：会先做验证再测试；配置仍从 `config/transformer_spoof19_wavlm.yaml` 读取。  
3. **仅 CPU / 其他 CUDA 版本**：需自行把 PyTorch 安装行换成 [pytorch.org](https://pytorch.org) 上对应命令，其余步骤不变。

若你希望 **「从全新 clone 到跑通」** 再缩短成最少条数，可以说一下你是否已有 **dump 好的 WavLM 特征**，我可以给一版「跳过特征导出」的最短指令链。