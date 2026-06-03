# 测试矩阵报告（T9）

## 生成方式

1. 安装开发依赖：`pip install -r requirements-dev.txt`
2. 运行矩阵脚本（JUnit + 环境快照 + Markdown/HTML）：

```bash
python scripts/run_full_test_matrix.py
```

产出目录默认为 `reports/`，包含：

- `test_report.md` / `test_report.html`：按任务标签 **T0–T9** 汇总的用例与耗时
- `junit.xml`：pytest JUnit 输出（含 `record_metric` 写入的 `<property name="metric_*"/>`）
- `env.json`：Python / torch / XPU / CUDA 等探测信息

## 任务与测试文件映射

逻辑见 `scripts/_report_writer.py` 中的 `TASK_BY_FILE`。新增测试文件时，请在该表中登记 `test_xxx -> Tn`，以便报告正确归类。

**录音端到端（E2E）**：`tests/test_e2e_by_feature.py` 映射为任务 **`E2E`**；功能点与前置条件见 **[FEATURE_E2E_TESTS.md](./FEATURE_E2E_TESTS.md)**。跑前请执行 `python scripts/generate_test_fixtures.py`。

## 本地快速 pytest

```bash
pytest tests/ -q
```

带指标并写 JUnit：

```bash
pytest tests/ --junitxml=reports/junit.xml
```

## Intel XPU

部分用例带 `@pytest.mark.xpu`，无 XPU 时会自动 skip。建议在 XPU 环境设置：

```bash
set PYTORCH_ENABLE_XPU_FALLBACK=1
```

（Windows PowerShell: `$env:PYTORCH_ENABLE_XPU_FALLBACK=1`）
