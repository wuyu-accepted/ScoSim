# 四维实验全生命周期 Usage 账本

本账本只统计正式实验的权威审计文件；pilot、握手测试、废弃分片和重复合并文件均不计入。金额因价格版本未冻结而统一记为 `NA_price_not_frozen`。

| 维度 / 实验 / 方法 | 成功网络调用 | 总 Token |
|---|---:|---:|
| dimension2 / cross_method / belief_engine | 2216 | 1427162 |
| dimension2 / cross_method / campuspulse | 512 | 1607362 |
| dimension2 / cross_method / chuang_opinion_dynamics | 512 | 876770 |
| dimension2 / cross_method / generative_agents | 1214 | 1201714 |
| dimension2 / cross_method / shared_cross_method | 3131 | 768009 |
| dimension2 / cross_method_blind_review / external_judge | 168 | 283666 |
| dimension2 / longitudinal_ablation / campuspulse_ablation_conditions | 1443 | 2788148 |
| dimension2 / longitudinal_ablation_blind_review / external_judge | 319 | 504186 |
| dimension3 / cross_method / belief_engine | 2002 | 1328881 |
| dimension3 / cross_method / chuang_et_al | 9000 | 22446251 |
| dimension3 / cross_method / generative_agents | 50740 | 46333161 |
| dimension3 / longitudinal_ablation / campuspulse_no_profile | 9000 | 26411710 |
| dimension3 / longitudinal_ablation / campuspulse_no_profile_no_smc | 6009 | 9245193 |
| dimension3 / longitudinal_ablation / campuspulse_single_particle | 9000 | 16416246 |
| dimensions1_and_4 / formal_paired_forum_run / campuspulse | 1126 | 6337189 |

完整的调用类型拆分见 `usage_by_method_and_call_type.csv`。缓存命中不重复计费。维度一/四只有交接包验证过的合计，维度二共享嵌入不能安全拆分到单个 baseline。
