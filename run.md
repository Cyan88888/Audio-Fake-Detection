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
4. **Web 上传音频**：服务端按扩展名白名单校验（见 `web/audio_formats.py`）；解码依赖 `torchaudio`。**MP3、AAC、M4A** 等一般需要系统安装 **FFmpeg**，否则会解码失败并返回 400。

若你希望 **「从全新 clone 到跑通」** 再缩短成最少条数，可以说一下你是否已有 **dump 好的 WavLM 特征**，我可以给一版「跳过特征导出」的最短指令链。

-------------------------------------------

已按你的要求精简 Web：现在只保留**基础音频检测**，删除/移除了历史记录、鉴权、审计、数据库相关代码链路。

## 已修改内容
后端现在只保留这些接口：

```text
GET  /
GET  /health
POST /api/predict
POST /api/predict_batch
GET  /api/tasks/{job_id}
```

已移除：

- `/api/auth/login`
- `/api/history`
- `/api/history/export.csv`
- `/api/history/export.json`
- `/api/audit`
- SQLite 数据库写入
- token 鉴权
- 审计日志
- 历史记录保存

已检查：

- 后端 Python 语法检查通过
- React 前端构建通过
- FastAPI 当前路由检查通过

## 当前报错原因
你终端里的：

```text
Waiting for application startup.
Killed
```

不是普通 Python 报错，而是进程被系统直接杀掉。结合当前终端信息：

```text
内存：2 GB
GPU ：No devices were found
```

主要原因基本可以确定是：**启动时加载 HuBERT 特征前端和检测模型占用内存过高，2GB 内存不足，系统 OOM 后直接 kill 进程**。

如果要使用 HuBERT 最优模型进行 Web 推理，建议至少换到有 GPU 或更高内存的实例。当前 2GB 内存环境不适合在线加载 HuBERT。

## HuBERT 最优模型启动命令
在资源足够的环境下运行：

```bash
cd /root/autodl-tmp/SafeEar
```

```bash
conda activate safeear
```

```bash
export SAFEAR_CKPT=Exps/Search_PoolMax_S3_ls002_hubert/checkpoints/epoch=4-val_eer=0.0009.ckpt
```

```bash
export SAFEAR_FEAT=hubert
```

```bash
export SAFEAR_HUBERT=model_zoos/hubert_base_ls960.pt
```

```bash
export SAFEAR_WEB_FIXED_THRESHOLD_SPOOF=0.5
```

```bash
uvicorn web.api:app --host 0.0.0.0 --port 8080
```

如果仍然 `Killed`，就是运行环境内存不足，不是代码逻辑错误。

## 当前前端页面路由
前端主要在 `web/frontend/src/App.jsx` 中控制页面：

```text
/          上传页面
/loading   检测中页面
/result    检测结果页面
```

实际逻辑是：

- 访问 `/` 或其他未匹配路径：显示上传页面；
- 访问 `/loading`：显示加载进度页面；
- 访问 `/result`：显示结果页面；
- 正常点击“开始检测”时，前端通过状态切换页面，不一定改变浏览器 URL。

你要调整页面布局，主要改这几个区域：

- 上传页：`currentPage === "upload"` 对应最后一个 `return`
- 加载页：`currentPage === "loading"`
- 结果页：`currentPage === "result"`

## 阈值建议
更推荐：**后端固定阈值**。

原因是你的系统是毕业设计展示和检测系统实现，不是阈值调参平台。后端固定阈值有几个优点：

- 结果更稳定，便于复现实验；
- 避免前端随意改阈值导致结果前后不一致；
- 论文中更容易说明系统判决规则；
- 更符合“部署系统”的设计思路。

当前代码已经采用后端固定阈值：

```bash
export SAFEAR_WEB_FIXED_THRESHOLD_SPOOF=0.5
```

前端即使传 `threshold`，后端也会忽略，统一使用环境变量中的固定阈值。