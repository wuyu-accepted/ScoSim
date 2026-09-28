# AAMAS2027 四维实验 Evidence v1

本目录按四个评估维度组织。四个维度均已封装可公开的最低层数据或充分统计量、指标结果、独立报告、参数、案例说明、费用账本和来源证明。维度二、三的论文展示值以 `publication_authority/Comparative_Experiment_Plan_FINAL.md` 为唯一权威。

全生命周期 usage 已按“方法 × 调用类型”聚合；各实验目录内提供过滤后的 `cost_by_method_and_call_type.csv`。由于模型价格版本未冻结，金额字段保持 `NA_price_not_frozen`，不推测费用。API Key、原始 Provider 响应和可恢复身份的信息不进入本发布包。

复现从 `REPRODUCE.md` 开始；`environment/` 冻结依赖，`verify_release.py` 提供零网络、零 API 的发布包验收。维度二横向目录中的 `independent_blind_review_template.csv` 是未填写的独立盲审模板，不是论文结果；正式盲评标签及第二模型复核统计分别见同目录 JSONL 与 metrics。
