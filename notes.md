建议你在这个 Web 系统里把固定阈值设为：

- **`SAFEAR_WEB_FIXED_THRESHOLD_SPOOF=0.115`**（四舍五入）

原因：
- 你当前最终模型 `Search_PoolMax_S3_ls002` 的校准结果里，
  - `test_calib_threshold_bonafide ≈ 0.885`
  - 所以 `threshold_spoof = 1 - 0.885 ≈ 0.115`
- 这是按 dev 校准得到的阈值，和你论文流程一致，比拍脑袋用 0.5 更合理。

---

如果你按业务偏好微调，可参考：

- **均衡默认（推荐）**：`0.115`
- **更严格防伪造（降低漏检伪造）**：`0.08 ~ 0.10`
- **更少误伤真音频（降低误报）**：`0.13 ~ 0.16`

---

设置方式：

```bash
export SAFEAR_WEB_FIXED_THRESHOLD_SPOOF=0.115
uvicorn web.api:app --host 0.0.0.0 --port 8080
```

如果你愿意，我可以再给你一个“一页式阈值说明”（论文+系统文档都能用），写清楚为什么选 0.115。


`test_calib_threshold_bonafide` 不是在 test 上“再训练”出来的，而是这样来的：

1. **先在验证集（dev）上扫阈值**  
   在训练/验证阶段，你的脚本会计算 `val_minDCF_th`（现在升级后优先 `val_min_tDCF_th`，没有再回退 `val_minDCF_th`）。

2. **取这个 dev 最优阈值作为 bonafide 阈值**  
   记为 `threshold_bonafide`。

3. **把它原样用于 test**  
   然后在 test 上按这个阈值重算 FAR/FRR/ACC/F1，并写入：
   - `test_calib_threshold_bonafide`
   - `test_calib_*`

所以本质是：**dev 定阈值，test 只评估**（避免 test 泄漏）。