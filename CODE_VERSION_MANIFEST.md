# 代码版本与可复现性边界

## 结论

论文实验代码与比赛提交版本**不完全相同**。比赛系统被保存在只读交接包中；`campuspulse_core` 是从交接包提取的最小研究内核，并为论文对比、消融、统一 Schema、真实模型调用、缓存恢复和统计评估做了版本化扩展。

因此，论文和仓库不得声称“实验代码与比赛 GitHub 最新版本完全未修改”。更准确的表述是：比赛系统作为来源快照保留，论文实验在其最小研究内核上实现，并通过冻结协议、文件哈希和 evidence 清单追踪。

## 当前冻结标识

| 对象 | 冻结标识 |
|---|---|
| 论文实验协议 | `locked_protocol.yaml`，版本 `1.3.0` |
| 协议 SHA-256 | `1cc0678d116204cae61c3c26c4002058a1a48cad71553474ebdb95e416a2b915` |
| 最小核心来源清单 | `campuspulse_core/SOURCE_MANIFEST.sha256.csv` |
| 来源清单 SHA-256 | `0807f728f907b1fa23d176a34bf7672266fcdf73602732470ef21e3423cc266d` |
| Generative Agents 上游 | `joonspk-research/generative_agents`，commit `fe05a71d3e4ed7d10bf68aa4eda6dd995ec070f4` |
| Chuang opinion dynamics 上游 | `yunshiuan/llm-agent-opinion-dynamics`，commit `edfd7c91c366307ee06344477dcc8082eb720215` |
| Belief Engine | 按论文机制进行本地适配性复现；当前没有冻结的作者源代码 commit |

## 论文阶段新增能力

- 四方法公平初始化和统一论坛/提示词边界；
- 统一 JSONL Schema 与方法适配器；
- 真实模型提供器、预算、重试、缓存和原始响应审计；
- Belief Engine、Generative Agents 与 Chuang 正式执行器；
- 维度二状态—表达指标、盲评、横向比较和四条件消融；
- 维度三共同群体重放、网络传播、横向比较和纵向机制消融；
- 普通矩 Sarle BC、Gaussian KDE 和 20 箱信息熵；
- 断点恢复、并发执行、只读复算、usage 聚合与 evidence 封装。

## 当前限制

`campuspulse_core` 目录本身不是 Git 工作树，因此当前论文代码不能用单一 commit 完整标识，必须以文件级 SHA-256 清单为准。发布前应在论文仓库建立正式 Git commit，并重新生成覆盖当前所有脚本、配置、Schema 和报告的 release manifest。
