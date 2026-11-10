# 2026-11-10 整理前快照与待办

先保存现状，再做一次整理提交。快照不表示这些实验已成为默认科研方法。
snapshot_inventory.json逐项保存源文件、SHA256和待整理类别。

1. 正式工具保留在src/scripts；独立OF验证移入tests并保留旧命令入口。
2. 论文/结果分析移入analysis；教学/机制探索放入专题experiments。
3. 两批scratch移到archive，5对完全重复代码合并，保留路径对照。
4. clean拟合/绘图开发盘源码原样复制到legacy_snapshot，下一提交拆分入口与分析。
5. 数据盘原始开发目录保持原样；不搬依赖包、波形、权重、结果、日志。
6. 旧入口提供兼容转发；补充AGENTS.md用途分层与实验记录规则。

待注意：warmstart调优表只判断能量，未判断1.382mV噪声档，不是通用推荐。
plot_efficiency.py的max_energy帮助文字与实际定标行为不一致，整理时修正文案。
旧科研假设/固定路径/过时结果保留说明，目录整理不悄悄改变算法。
本次仅本地commit；不push、不fetch/pull、不合并上游。
