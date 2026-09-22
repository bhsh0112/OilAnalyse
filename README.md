# 营运车辆油料消耗分析

课程作业：**交通信息采集与集成 · 第一次大作业**。

用油耗仪 + 车载定位轨迹，在无加油标签的条件下检出加油事件，再核算作业要求的四项数字。分析是离线流水线，产物是带图 Markdown 报告，不是 Web 应用。

| 题设指标 | 本数据结论（高可信 + 较可信） |
| --- | ---: |
| 加油次数 | **6** |
| 加油量 | **1002.2 L** |
| 耗油量 | **785.4 L** |
| 加油时间点 | 6 个 `GPSTime` 区间，见报告第 3 章 |

> **报告示例：** [`outputs/report.md`](outputs/report.md)  
> 用 VS Code 或 Typora 打开。图是相对路径 `figures/`；克隆后若配图缺失，在本地跑一次流水线即可生成。

车辆：吉 A8K650（终端 `simNo=18405190226`），2014-11-01 ～ 11-09，13128 条采样，约 3048.5 km。

---

## 方法摘要

1. **加油事件检测**：停车（≤ 5 km/h、位移 ≤ 300 m）+ 油量连续上升（单步 > 1 L、间隔 < 3 min 合并、体积 ≥ 25 L）。
2. **AD 分层**：只评油量上升段。油升且 AD 降为同向；段内 AD ≥ 1500 为尖峰；前一条尖峰不单独否决。输出高可信 / 较可信 / 存疑。
3. **空间旁证**：检索周边加油站，近站不上调存疑，无站不下调高可信。现势 POI 对照 2014 年轨迹，只印证、不定罪。
4. **油量平衡**：耗油 = 期初油量 + 总加油量 − 期末油量。不用负向 Δ油量积分。

管理核算只收 **高可信 + 较可信**；存疑进核查表。本批 12 个候选里 6 次入账、6 次存疑。

```text
xls 轨迹 ──► 清洗 / 噪声 σ ──► 加油事件检测（含 AD + 加油站旁证）
                                      │
                                      ▼
                               带可信度的事件表
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              题设四项            日台账 / 轨迹      管理建议（可选）
         次数 · 加油量 · 耗油 · 时间点
```

---

## 快速开始

Python 3.9+（Windows 下绘图使用微软雅黑）。

```bash
pip install -r requirements.txt
```

把作业下发的 Excel 放到 `data/`，然后在仓库根目录执行：

```bash
python -m src.fuel_mgmt.pipeline --data "data/2-油耗检测作业-吉A8K65011月1日-10日数据.xls" --out outputs
```

| 产物 | 说明 |
| --- | --- |
| [`outputs/report.md`](outputs/report.md) | 完整分析报告（示例已入库） |
| `outputs/figures/` | 报告配图与高德轨迹 HTML |
| `outputs/analysis_summary.json` | 给大模型用的结构化事实，不含 1 万余行原始点 |

不调用大模型、不检索加油站：

```bash
python -m src.fuel_mgmt.pipeline --data "data/2-油耗检测作业-吉A8K65011月1日-10日数据.xls" --out outputs --skip-llm --skip-poi
```

```bash
python -m pytest tests -q
```

---

## 可选接口

密钥不要写入代码，也不要提交 git。文件放在 `api/`（已 gitignore），或写进 `.env`。

### DeepSeek（报告第 5 章：油量管理建议）

次数、加油量、耗油、时间点由规则算出，模型只根据摘要写建议，不得改数字。

- 文件：`api/deepseek-api.txt`（单行 Key）
- 或环境变量：`DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL=deepseek-chat`

未配置时第 5 章标注未生成，前四章不受影响。

### 高德（空间旁证）

- 文件：`api/gaode-api.txt`
- 需要 **Web 服务** Key。JS 端 Key 会返回 `USERKEY_PLAT_NOMATCH`，流水线回退 OpenStreetMap Nominatim。
- 空间距离只作旁证，不否决 AD 结论。

---

## 目录

```text
.
├── README.md
├── requirements.txt
├── pytest.ini
├── api/                    # 本地密钥（不入库）
├── data/                   # 原始 xls（不入库）
├── src/fuel_mgmt/          # 分析包
│   ├── pipeline.py         # 一键入口
│   ├── load.py / quality.py
│   ├── refuel.py           # 初版检测 + 油量平衡
│   ├── evidence.py         # AD 分层、陡降
│   ├── stations.py         # 加油站旁证
│   ├── mining.py           # 日台账、时机、间隔油耗
│   ├── report.py / visualize.py
│   └── llm_advisor.py      # DeepSeek 建议
├── tests/
└── outputs/
    └── report.md           # 报告示例（入库）
```

`outputs/` 下其余运行产物（配图缓存、JSON、汇报稿）默认不入库。
