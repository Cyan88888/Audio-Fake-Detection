# SafeEar 实验配置与指标总表（自动生成）

**数据来源**：`Exps/<实验名>/config.yaml`、`test_results.json`。无 `test_results.json` 表示尚未跑 `test.py` 或未归档。

**列名缩写**：L/H=Transformer层数/头数；PE=位置编码；pool=池化；ls=label smoothing；tm_p/T=time mask 概率/最大帧数；tta_n/fr=TTA段数/每段帧长；gclip=梯度裁剪；eval_full=测试时是否返回全长特征；online_train_WavLM=训练是否在线提WavLM特征。

**特殊说明**：`Cross2021_from_Ablation_S2` 为 ASVspoof2021 数据配置；若 **min-tDCF 为 nan**，通常表示当前任务未加载或未匹配 2021 所需的 ASV 分数文件（与 2019 LA 的 min-tDCF 计算条件不同），此时以 **EER** 等仍有效的指标为主。

**基线 `TransformerSpoof19_wavlm_e30`**（WavLM + 2L/8H/sine、dropout 0.1，与 S1 消融前的默认配置）：**val/test EER、test minDCF、ROC-AUC** 摘自该实验目录下历史 `log.txt`（与当前 `test_results.json` 口径一致的训练结束评测）；当时日志未打印 **min-tDCF**（列中填 `-`），若需与消融行对齐，请用含 `system.use_min_tdcf: true` 的配置对同一 checkpoint 运行 `test.py` 并归档 `test_results.json`。

## 表1 配置与结果总表

| 实验目录 | train_feat | online_train_WavLM | batch | epochs | lr | L | H | drop | PE | pool | ls | tm_p | tm_T | tta_n | tta_fr | gclip | eval_full | val_EER | val_min-tDCF | test_EER | test_min-tDCF | test_minDCF | test_ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ablation_Pool_max | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 2 | 16 | 0.15 | learnable | max | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| Ablation_Pool_mean | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 2 | 16 | 0.15 | learnable | mean | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| Ablation_Pool_meanmax | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 2 | 16 | 0.15 | learnable | meanmax | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| Ablation_S1_l2_h16_d015 | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 2 | 16 | 0.15 | sine | attention | - | - | - | - | - | - | - | 0.032546 | 0.094087 | 0.081306 | 0.200953 | 0.030265 | 0.976429 |
| Ablation_S1_l3_d015 | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 3 | 8 | 0.15 | sine | attention | - | - | - | - | - | - | - | 0.045923 | 0.113837 | 0.091084 | 0.207794 | 0.036096 | 0.968994 |
| Ablation_S1_l4_d020 | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 4 | 8 | 0.2 | sine | attention | - | - | - | - | - | - | - | 0.046697 | 0.132280 | 0.091207 | 0.212667 | 0.035384 | 0.969693 |
| Ablation_S2_pe_learnable | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 2 | 16 | 0.15 | learnable | attention | - | - | - | - | - | - | - | 0.024719 | 0.070492 | 0.061725 | 0.156753 | 0.023891 | 0.985259 |
| Ablation_S2_pe_learnable_TTA1 | …/WavLM_base/…/train | - | 32 | 1 | 0.0003 | 2 | 16 | 0.15 | learnable | attention | - | - | - | 1 | 201 | - | True | 0.024719 | 0.070492 | 0.063901 | 0.164801 | 0.025349 | 0.983812 |
| Ablation_S2_pe_learnable_TTA3 | …/WavLM_base/…/train | - | 32 | 1 | 0.0003 | 2 | 16 | 0.15 | learnable | attention | - | - | - | 3 | 201 | - | True | 0.024719 | 0.070492 | 0.064446 | 0.164095 | 0.024676 | 0.984536 |
| Ablation_S2_pe_learnable_TTA5 | …/WavLM_base/…/train | - | 32 | 1 | 0.0003 | 2 | 16 | 0.15 | learnable | attention | - | - | - | 5 | 201 | - | True | 0.024719 | 0.070492 | 0.064715 | 0.165056 | 0.024670 | 0.984462 |
| Ablation_S2_pe_sine | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 2 | 16 | 0.15 | sine | attention | - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ablation_S3_aug_mild | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 2 | 16 | 0.15 | learnable | attention | 0.02 | 0.2 | 8 | - | - | - | - | - | - | - | - | - | - |
| Ablation_S3_aug_time_mask | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 2 | 16 | 0.15 | learnable | attention | 0 | 0.5 | 20 | - | - | - | - | 0.027483 | 0.082192 | 0.063193 | 0.170156 | 0.026022 | 0.983129 |
| Ablation_S3_aug_time_mask_ls005 | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 2 | 16 | 0.15 | learnable | attention | 0.05 | 0.5 | 20 | - | - | - | - | - | - | - | - | - | - |
| Ablation_S3_aug_time_mask_ls005_clip1 | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 2 | 16 | 0.15 | learnable | attention | 0.05 | 0.5 | 20 | - | - | 1 | - | - | - | - | - | - | - |
| Ablation_S4_codec_offline | …/WavLM_base_aug/…/train_mix | - | 64 | 30 | 0.0003 | 2 | 16 | 0.15 | learnable | attention | - | - | - | - | - | 1 | - | 0.027791 | 0.085232 | 0.061725 | 0.156246 | 0.024244 | 0.985806 |
| Ablation_S4_online_codec_quick | …/WavLM_base/…/train | True | 16 | 5 | 0.0003 | 2 | 16 | 0.15 | learnable | attention | - | - | - | - | - | 1 | - | - | - | - | - | - | - |
| Cross2021_from_Ablation_S2 | …/WavLM_base/…/train | - | 64 | 1 | 0.0003 | 2 | 16 | 0.15 | learnable | attention | - | - | - | - | - | - | - | 0.144643 | nan | 0.144643 | nan | 0.035329 | 0.910239 |
| TransformerSpoof19_hubert_e30 | …/ASVSpoof2019_Hubert_L9/…/train | - | 64 | 30 | 0.0003 | 2 | 8 | 0.1 | sine | attention | - | - | - | - | - | - | - | - | - | - | - | - | - |
| TransformerSpoof19_wavlm_e30 | …/WavLM_base/…/train | - | 64 | 30 | 0.0003 | 2 | 8 | 0.1 | sine | attention | - | - | - | - | - | - | - | 0.033785 | - | 0.082935 | - | 0.037304 | 0.971788 |

## 表2 测试 checkpoint 路径

| 实验目录 | ckpt_path |
| --- | --- |
| Ablation_Pool_max | `-` |
| Ablation_Pool_mean | `-` |
| Ablation_Pool_meanmax | `-` |
| Ablation_S1_l2_h16_d015 | `Exps/Ablation_S1_l2_h16_d015/checkpoints/epoch=20-val_eer=0.0325.ckpt` |
| Ablation_S1_l3_d015 | `Exps/Ablation_S1_l3_d015/checkpoints/epoch=1-val_eer=0.0459.ckpt` |
| Ablation_S1_l4_d020 | `Exps/Ablation_S1_l4_d020/checkpoints/epoch=1-val_eer=0.0467.ckpt` |
| Ablation_S2_pe_learnable | `Exps/Ablation_S2_pe_learnable/checkpoints/epoch=7-val_eer=0.0247.ckpt` |
| Ablation_S2_pe_learnable_TTA1 | `Exps/Ablation_S2_pe_learnable/checkpoints/epoch=7-val_eer=0.0247.ckpt` |
| Ablation_S2_pe_learnable_TTA3 | `Exps/Ablation_S2_pe_learnable/checkpoints/epoch=7-val_eer=0.0247.ckpt` |
| Ablation_S2_pe_learnable_TTA5 | `Exps/Ablation_S2_pe_learnable/checkpoints/epoch=7-val_eer=0.0247.ckpt` |
| Ablation_S2_pe_sine | `-` |
| Ablation_S3_aug_mild | `-` |
| Ablation_S3_aug_time_mask | `Exps/Ablation_S3_aug_time_mask/checkpoints/epoch=7-val_eer=0.0275.ckpt` |
| Ablation_S3_aug_time_mask_ls005 | `-` |
| Ablation_S3_aug_time_mask_ls005_clip1 | `-` |
| Ablation_S4_codec_offline | `Exps/Ablation_S4_codec_offline/checkpoints/epoch=9-val_eer=0.0278.ckpt` |
| Ablation_S4_online_codec_quick | `-` |
| Cross2021_from_Ablation_S2 | `Exps/Ablation_S2_pe_learnable/checkpoints/epoch=7-val_eer=0.0247.ckpt` |
| TransformerSpoof19_hubert_e30 | `-` |
| TransformerSpoof19_wavlm_e30 | `Exps/TransformerSpoof19_wavlm_e30/checkpoints/epoch=17-val_eer=0.0338.ckpt` |