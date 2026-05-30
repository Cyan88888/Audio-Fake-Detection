# AudioForgeryDet

基于 HuBERT 帧特征与 Transformer 检测头的音频伪造检测系统（本科毕设实验代码）。

检测流程：自监督前端（HuBERT / WavLM / wav2vec 2.0）→ 帧级特征 → `FrameTransformerDetector` → 伪造/真实分类。

## 环境

```bash
pip install -r requirements.txt
```

可选环境变量（新前缀 `SPOOFDET_*`，仍兼容 `SAFEAR_*`）：

- `SPOOFDET_ASVSPOOF2019_ROOT`：ASVspoof 2019 LA 数据根目录
- `SPOOFDET_CKPT` / `SPOOFDET_FEAT` / `SPOOFDET_HUBERT`：推理与 Web 部署

## 训练与测试

```bash
python train.py --conf_dir config/ablation_supplement/m0_base.yaml
python test.py --conf_dir config/ablation_supplement/m0_base.yaml \
  --ckpt_path Exps/Ablation_Hubert_pool_mean/checkpoints/epoch=7-val_eer=0.0012.ckpt
```

## Web 推理

```bash
./start_web.sh
```

## 项目结构

- `spoofdet/`：数据模块、检测模型、训练器、评测指标
- `config/`：实验配置（Hydra `_target_: spoofdet.*`）
- `inference/`：离线推理与特征提取
- `web/`：FastAPI + React 演示界面
- `Exps/`：实验输出与 checkpoint

