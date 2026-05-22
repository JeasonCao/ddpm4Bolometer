# DDPM Training Report

## Run Summary

| 项目 | 内容 |
|------|------|
| 开始时间 | 2026-05-21 |
| 停止时间 | 2026-05-22 ~17:xx（epoch 53 后手动停止） |
| 总训练轮次 | 53 / 100 epochs |
| 模型架构 | UNet1D，~15.1M 参数 |
| 损失函数 | L1 noise prediction |
| 扩散步数 | T = 50，quadratic beta schedule |
| batch size | 16，混合精度（float16 AMP） |
| 学习率 | 2e-4，CosineAnnealingLR |
| 训练集 | 81,450 样本（clean × noise 随机配对） |
| 验证集 | 9,050 样本 |
| 硬件 | RTX 3060 12GB，WSL2 |
| 每 epoch 用时 | ~29–36 分钟 |

---

## Loss 曲线

| Epoch | Train Loss | Val Loss | 备注 |
|-------|-----------|---------|------|
| 1     | 0.032521  | 0.025332 | |
| 5     | 0.014488  | 0.015707 | |
| 10    | 0.011263  | 0.012087 | |
| 20    | 0.008794  | 0.009051 | |
| 30    | 0.007154  | 0.006441 | |
| 40    | 0.006061  | 0.005789 | |
| 49    | —         | 0.004800 | **最佳 val_loss** → 已保存为 best_model.pt |
| 50    | 0.005091  | 0.005292 | checkpoint_050.pt 保存 |
| 51    | 0.004958  | 0.005001 | |
| 52    | 0.004888  | 0.005801 | |
| 53    | 0.004806  | 0.005218 | ← 本次停止点 |

**观察：**
- Val loss 在 epoch 49 达到最低 0.004800，之后出现轻微波动（轻度过拟合迹象）
- Train loss 仍在缓慢下降，但 val loss 已基本收敛
- `best_model.pt` 对应 epoch 49，是评估和推理应使用的版本

---

## 输出文件

```
/home/wsl_0vbb/DDPM4bolometer/model_output/
├── best_model.pt          ← 最佳模型（epoch 49，val_loss=0.004800）★ 用这个
├── checkpoint_010.pt      ← epoch 10 完整 checkpoint（含 optimizer state）
├── checkpoint_020.pt      ← epoch 20
├── checkpoint_030.pt      ← epoch 30
├── checkpoint_040.pt      ← epoch 40
├── checkpoint_050.pt      ← epoch 50 ← 续训起点
├── history.json           ← 完整 loss 历史
├── loss_curve.png         ← loss 曲线图
└── config.json            ← 训练配置
```

---

## 如何续训

如需继续从 epoch 50 训练到 100（使用 checkpoint_050.pt 恢复 optimizer 状态）：

```bash
cd /mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#

/home/wsl_0vbb/mambaforge3/envs/DDPM/bin/python -u -m src.ddpm.train \
    --clean_dir  /home/wsl_0vbb/DDPM4bolometer/simu_data/clean \
    --noise_dir  /home/wsl_0vbb/DDPM4bolometer/simu_data/noise \
    --output_dir /home/wsl_0vbb/DDPM4bolometer/model_output \
    --resume     /home/wsl_0vbb/DDPM4bolometer/model_output/checkpoint_050.pt \
    --loss l1 --epochs 100 --batch_size 16 --lr 2e-4 \
    --num_workers 4 --save_every 10 --amp
```

**注意：**
- `--resume` 必须接 `checkpoint_XXX.pt`（含 optimizer state），**不能**接 `best_model.pt`
- `--epochs 100` 是总 epoch 数，脚本会自动从 epoch 51 继续
- 续训完成后 `best_model.pt` 会在 val_loss 有改善时自动更新

---

## 评估建议

目前 epoch 49 的模型（`best_model.pt`）已足够用于评估。  
建议直接进入后训练分析流程，续训可在有需要时再进行。
