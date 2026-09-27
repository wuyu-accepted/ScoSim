# 本维度代码快照

本目录保存生成、评估或复核该维度结果所需的整理后源码副本。文件保持其在 `campuspulse_core/` 下的相对路径，便于审计和回放。

- 这些文件是发布快照，不是新的开发主目录；修改应回到 `campuspulse_core/` 后重新运行封装脚本。
- `SOURCE_MANIFEST.json` 记录每个源文件的原路径、SHA-256、字节数和快照路径。
- 共享执行器、统一 Schema 与适配器会在需要它们的维度中各自收录，以使单个维度材料自足。
- 不包含 API Key、Provider 原始响应、缓存、运行日志、身份映射或 `__pycache__`。
- 结果展示值仍以 `publication_authority/Comparative_Experiment_Plan_FINAL.md` 为准。
