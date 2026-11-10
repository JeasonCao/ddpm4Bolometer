# 项目代码层级

| 位置 | 用途 |
|---|---|
| src/ | 可复用算法和模型 |
| scripts/ | 正式入口：OF、生成、训练、推理、拟合 |
| tests/ | 短验证 |
| analysis/ | 已有结果的指标和科研/论文图 |
| experiments/ | 围绕研究问题的脚本、配置、说明 |
| archive/ | 历史试验，非当前推荐方法 |
| scratch/ | 临时探索，默认不提交 |

实验代码值得commit；commit不代表它成为默认科研方法。agent按用途选目录，按证据与验证判断是否升级，不按作者分类。

数据/环境留数据盘；日记、TODO、图及fit HDF5沿用Obsidian约定。不要将_development的依赖、日志、锁文件和结果整体搬进仓库。

整理前快照：8e8e3e6。旧scripts入口大多保留小型转发文件，源码只有一份；之后改canonical文件。详见 [整理记录](reorganization_20261110/README.md)。本次只整理此前未提交的研究工作，既有主线目录不全面迁移。
