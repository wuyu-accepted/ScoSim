# 冻结运行环境

- `requirements.txt`：统计、Schema 与科学计算依赖。
- `requirements-real-provider.lock.txt`：真实模型 Provider 的精确锁定依赖。
- `PYTHON_VERSION.txt`：本次验证使用的解释器版本。

金额和模型单价不属于软件环境；未冻结价格继续记为 `NA_price_not_frozen`。
