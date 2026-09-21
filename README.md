# 营运车辆油耗在线监测（第一次大作业）

基于油耗仪 + GPS 轨迹，离线分析单车加油/耗油，并生成带图 Markdown 报告。

## 环境

Python 3.9+（已在 Windows 下验证中文字体 `Microsoft YaHei`）。

```text
pip install -r requirements.txt
```

## 生成报告

在作业根目录执行：

```text
python -m src.fuel_mgmt.pipeline --data "data/2-油耗检测作业-吉A8K65011月1日-10日数据.xls" --out outputs
```

产物：

- `outputs/report.md`：完整分析报告（请用 VS Code / Typora 预览，图片为相对路径）
- `outputs/figures/*.png`
- `outputs/analysis_summary.json`：给大模型用的结构化事实，不含原始 1 万余行

不调用大模型：

```text
python -m src.fuel_mgmt.pipeline --data "data/2-油耗检测作业-吉A8K65011月1日-10日数据.xls" --out outputs --skip-llm
```

## DeepSeek（可选，第 5 章管理建议）

使用 DeepSeek 的 OpenAI 兼容接口。不要把 Key 写进代码或提交到 git。

任选一种方式提供密钥：

1. 作业根目录的 `deepseek-api.txt`（单行 Key；已 gitignore）
2. 环境变量 / `.env`：

```text
DEEPSEEK_API_KEY=sk-你的密钥
DEEPSEEK_MODEL=deepseek-chat
```

未配置 Key 时，报告第 5 章会注明「未生成」，前四章不受影响。

## 测试

```text
python -m pytest tests -q
```

## 目录

- `data/` 原始 xls（不要改）
- `src/fuel_mgmt/` 分析流水线
- `tests/` 合成序列单测
- `outputs/` 运行产物
