# tools/

`tools/` 只放**临时检查、调试、一次性工具脚本**。

- `tools/` 中的脚本**不应**被 `train.py`、`eval.py`、`infer.py` 依赖。
- `tools/` 中的文件理论上可以删除，**不影响主流程复现**。
- **稳定功能**应放入 `utils/`。
- **稳定运行流程**应放入 `scripts/`。
