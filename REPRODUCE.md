# 复现与只读核验指南

## 1. 环境

推荐使用 Python 3.14。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r environment\requirements.txt
python -m pip install -r environment\requirements-real-provider.lock.txt
```

只读统计复算不需要 API Key。真实模型重跑需要自行提供兼容 OpenAI API 的 `base_url` 和 API Key；凭据不得写入仓库。

## 2. 代码布局

每个维度的 `code/campuspulse_core/` 保持原研究内核的相对路径。运行该维度代码前，将以下目录加入 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = "$PWD\02_micro_consistency\code\campuspulse_core;$PWD\02_micro_consistency\code\campuspulse_core\src"
```

维度三使用相同方式，将路径中的 `02_micro_consistency` 改为 `03_population_polarization`。

## 3. 无费用复现边界

- 维度一与四：可从已发布充分统计量只读核验，不进行模型调用。
- 维度二与三：仓库发布最低层记录、评分标签、群体状态与正式指标；可运行统计和完整性检查。
- 真实生成重跑会产生费用，且依赖未公开的 Provider 凭据；它不是验证论文表格所必需的步骤。
- 论文最终展示值以 `publication_authority/Comparative_Experiment_Plan_FINAL.md` 为唯一权威。

## 4. 零调用验收

```powershell
python verify_release.py
```

成功时输出 `status: pass`，并验证：总清单、六个实验 provenance、代码快照、必需文件、空白盲审模板命名，以及敏感文件名隔离。

## 5. 各维度统计入口

| 维度 | 主要代码入口 | 已发布输入/输出 |
|---|---|---|
| 维度一 | `01_forum_realism/code/campuspulse_core/scripts/recompute_dimensions_1_4_readonly.py` | `01_forum_realism/housing_allocation/` |
| 维度二横向 | `02_micro_consistency/code/campuspulse_core/scripts/render_dimension2_cross_method_report.py` | `02_micro_consistency/dimension2_cross_method/` |
| 维度二消融 | `02_micro_consistency/code/campuspulse_core/scripts/render_dimension2_longitudinal_report.py` | `02_micro_consistency/dimension2_ablation/` |
| 维度三横向 | `03_population_polarization/code/campuspulse_core/evaluation/dimension3.py` | `03_population_polarization/dimension3_cross_method/` |
| 维度三消融 | `03_population_polarization/code/campuspulse_core/scripts/report_dimension3_ablation.py` | `03_population_polarization/dimension3_ablation/` |
| 维度四 | `04_governance_response/code/campuspulse_core/scripts/recompute_dimensions_1_4_readonly.py` | `04_governance_response/housing_natural_vs_D/` |

正式执行器保留在相应 `code/` 目录中；运行前必须阅读冻结配置和 `parameters.json`，不得擅自改变种子、时间点或指标口径。
