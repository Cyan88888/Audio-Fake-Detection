# HuBERT 补充消融 — 一键过夜跑

## 已改代码

- `safeear/models/safeear.py`：`positional_embedding: none` 时不再加 PE
- `safeear/models/detector_bilstm.py`：新增 BiLSTM 检测器（HuBERT 特征 B,C,T）

## 配置文件（相对 M0 只改一项）

| 文件 | 实验目录 | 改动 |
|------|----------|------|
| `config/ablation_supplement/no_pe.yaml` | `Ablation_Hubert_supp_no_pe` | PE → none |
| `config/ablation_supplement/heads8.yaml` | `Ablation_Hubert_supp_heads8` | num_heads → 8 |
| `config/ablation_supplement/layers1.yaml` | `Ablation_Hubert_supp_layers1` | num_layers → 1 |
| `config/ablation_supplement/layers3.yaml` | `Ablation_Hubert_supp_layers3` | num_layers → 3 |
| `config/ablation_supplement/bilstm.yaml` | `Ablation_Hubert_supp_bilstm` | BiLSTM 检测头 |

母版参考：`config/ablation_supplement/m0_base.yaml`（= 已有 `Ablation_Hubert_pool_mean`）

## 一键运行（去睡觉）

**no_pe 已训+测完** 后，只跑剩余 4 组：

```bash
cd /root/autodl-tmp/SafeEar
conda activate safeear
chmod +x scripts/run_remaining_supplement.sh
mkdir -p Exps/_logs/supplement_ablations
nohup bash scripts/run_remaining_supplement.sh >> Exps/_logs/supplement_ablations/nohup.log 2>&1 &
```

（勿再跑旧版全量脚本从头 no_pe，除非删掉 no_pe 的 test_results.json。）

看进度：

```bash
tail -f Exps/_logs/supplement_ablations/nohup.log
```

跑完后看汇总：

```bash
grep -E "Ablation_Hubert_supp" Exps/_summary/all_test_results.md
```

## 单条手动（调试）

```bash
python train.py --conf_dir config/ablation_supplement/no_pe.yaml
python test.py --conf_dir config/ablation_supplement/no_pe.yaml \
  --ckpt_path Exps/Ablation_Hubert_supp_no_pe/checkpoints/<best>.ckpt
```

预计 5 组 ×（训练+测试）≈ 3～6 小时（串行，视 early stop 而定）。

## 先测 BiLSTM 会不会报错（推荐）

```bash
cd /root/autodl-tmp/SafeEar
conda activate safeear

# 最快：假数据，不读 npy（只验证新 BiLSTM + training_step）
python scripts/smoke_test_bilstm.py --synthetic-only

# 更完整：读 1 个真实 batch（需 GPU/足够内存）
python scripts/smoke_test_bilstm.py

# 通过后再跑完整训练
python train.py --conf_dir config/ablation_supplement/bilstm.yaml
```
