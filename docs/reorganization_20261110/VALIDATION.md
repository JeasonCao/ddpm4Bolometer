# 整理验证（2026-11-10）

范围：工程迁移检查，不是重新验证论文结论。未训练、未运行GPU推理、未全量重拟合。已有数据与结果作为只读输入。

## 通过的检查

- 162个迁移相关Python文件语法解析通过。
- 独立窗口OF已有测试：新tests入口和旧scripts入口均通过。
- 新旧拟合、OF生成、Bessel分析入口及clean监督/绘图、论文拼图、效率脚本的CLI帮助检查通过。
- 一个合成单脉冲经原冻结fitter拟合，relative RMS < 1e-3；独立worker进程返回成功；Windows独占锁拒绝重复占用。只写临时目录。
- 读取历史完成的fit结果，重建蓝色单事件筛选：61225个，source_shard/window_idx与历史selected_windows.npz逐项完全相同。
- 90条迁移映射、85个唯一目标均存在，兼容入口指向匹配；配置模板保留兼容副本。
- 42条历史archive映射与快照内容一致（忽略文本换行），含5对去重来源。
- 数据盘6个clean Python源码、README和3个已迁入仓库的OF文件共10个原件未改动。
- clean拟合17个核心函数/类AST与快照一致，包括fit_one、worker、Workers、copy_metadata、verify、fit_file、choose_pilot和summarize。仅main/note调整路径、锁和日记。
- clean绘图的read_all/quality/绘图函数、蓝色/3D/能量分析函数AST保持一致；plot_after_fit仅启动、日记、render末尾旧TODO自动替换调整。
- 11个迁移相关shell文件通过Git Bash `bash -n`，并以LF保存；.gitattributes约束后续checkout。

## 限制与已知问题

- 历史plot_template_peak_linearity没有--help解析；尝试该参数实际进入原WSL固定输入路径，因本机该路径无数据在画图前失败。没有产出或覆盖图；未把此脚本标为跨平台验证通过。
- archive/部分OF与教学脚本仍依赖历史数据路径、外部Octopus或WSL。本次做语法/来源核验，不重跑这些历史实验。
- POSIX锁仅代码检查，未做Linux运行验证；Windows锁和子进程已验证。
- 不以函数相同、solver成功或语法通过证明科学结论。warmstart噪声条件、baseline标签及历史单位解释仍见README中的待决项。
- 不强制续跑旧clean manifest；整理后源码哈希变化会拒绝混用。原数据盘运行目录保留。

## 本地验证证据

完整迁移映射见migration_map.json，原始来源/哈希见snapshot_inventory.json。此次检查脚本和输出保留在数据盘 `_development/repo_cleanup_20261110/`（validate.py、test_fit_relocation.py、final_audit.py、validation.json、final_audit.json）；它们是本机整理操作证据，不是科研生产入口。OF测试在仓库tests/test_window_of.py，可用 `python -m unittest tests.test_window_of` 运行。
