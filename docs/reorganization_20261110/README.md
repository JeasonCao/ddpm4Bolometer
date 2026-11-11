# 2026-11-10 代码整理记录

第一提交8e8e3e6保存现状，第二提交整理。snapshot_inventory.json是逐文件原始哈希/来源；migration_map.json是新旧路径/兼容入口/去重映射。快照中的needs_organization表示当时待办，不是当前完成状态。

完成组织：正式工具保留src/scripts；OF测试放tests；分析放analysis；讲解与OF流程放experiments；历史scratch放archive，去重5对。clean runner放experiments/train_clean_20261107，画图放analysis/train_clean，scripts/fit_train_clean.py作入口。新增AGENTS.md规范。16个pyc停止Git跟踪，本地不删；运行日志/锁/草稿忽略，科研JSON保留。

路径配置与依赖环境分离；旧入口转发兼容。仅修正plot_efficiency.py帮助文案，不改定标计算。数据盘旧开发目录原样保留作运行证据；之后修改仓库canonical源码，避免双向维护两份。

明确限制：

- run_batch_inference_compare.py调优表只判断能量、未判断1.382mV噪声档。未在整理中悄悄改变旧推理行为；通用化需另修/验证。
- baseline_resolution_kev标签公式/1.382mV物理含义仍是科研待决事项。
- 历史archive/教学/OF脚本仍有WSL路径、临时环境或外部Octopus依赖，不保证旧实验都可跨机器直接跑。
- 旧能量换算、fp16及残差均值/方差解释问题见archive/README.md和日记。
- clean runner仍是特定数据集实验，未宣称通用化。
- 未执行GPU训练或全量科研重跑。实际验证见VALIDATION.md。
- 本次仅本地commit，不push、不fetch/pull、不合并yunshan更新。
