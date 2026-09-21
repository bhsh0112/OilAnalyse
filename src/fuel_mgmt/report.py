"""将分析结果拼成带图 Markdown 报告。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .refuel import RefuelEvent


def render_report(
    *,
    summary: dict[str, Any],
    events: list[RefuelEvent],
    figures: dict[str, str],
    advice: str | None,
    threshold_l: float,
) -> str:
    """生成完整中文 Markdown 报告文本。

    Args:
        summary: 结构化摘要。
        events: 加油事件。
        figures: 逻辑名到相对路径。
        advice: DeepSeek 生成的建议；None 或错误占位则写未生成说明。
        threshold_l: 加油量阈值。

    Returns:
        Markdown 字符串。
    """
    v = summary["vehicle"]
    sensor = summary["sensor"]
    hw = summary["homework"]
    ops = summary["operations"]
    quality_noise = sensor["oil_noise_sigma_l"]

    parts: list[str] = []
    parts.append("# 营运车辆油料消耗分析报告")
    parts.append("")
    parts.append("车辆 SIM `{}`，分析时段 {} ～ {}，共 {} 条记录，累计行驶约 **{} km**。".format(
        v.get("sim_no"),
        v.get("time_start"),
        v.get("time_end"),
        v.get("n_records"),
        v.get("mileage_km"),
    ))
    parts.append("")
    parts.append("数据文件：`{}`".format(v.get("data_file")))
    parts.append("")
    parts.append("本报告由离线分析流水线自动生成。加油次数、加油量、耗油量由规则统计得到；第 5 章管理建议在配置 DeepSeek API 后生成。")
    parts.append("")

    parts.append("## 1. 任务与数据说明")
    parts.append("")
    parts.append("作业要求基于油耗仪 + 车载 GPS 轨迹，估算：**加油次数、加油量、耗油量、加油时间点**。")
    parts.append("")
    parts.append("处理约定：")
    parts.append("")
    parts.append("- 主时间轴使用 `GPSTime`，不使用平台接收时间 `recTime`（后者存在明显滞后）。")
    parts.append("- `mileageValue` 按 0.1 km 计数换算为公里。")
    parts.append("- `oilValue` 上限 320 L，视为油箱容积或传感器饱和值。")
    parts.append("- 耗油使用油量平衡，不用短时负向差分直接求和。")
    parts.append("")

    rec = sensor.get("rec_time_lag_min") or {}
    samp = sensor.get("sampling") or {}
    parts.append("## 2. 数据质量与油量噪声")
    parts.append("")
    parts.append("### 2.1 采样与时间轴")
    parts.append("")
    parts.append(
        "采样间隔中位数约 **{:.0f} s**，99 分位 {:.0f} s，最大间隔 {:.0f} s；"
        "超过 2 分钟的间隔 {} 次，超过 1 小时 {} 次。".format(
            samp.get("median_s") or 0,
            samp.get("p99_s") or 0,
            samp.get("max_s") or 0,
            samp.get("gap_gt_120s") or 0,
            samp.get("gap_gt_3600s") or 0,
        )
    )
    parts.append("")
    parts.append(
        "`recTime` 相对 `GPSTime` 的滞后：中位数 {:.2f} 分钟，最大 {:.1f} 分钟。"
        "因此加油时间点一律取 GPS 时刻。".format(
            rec.get("median") or 0,
            rec.get("max") or 0,
        )
    )
    parts.append("")
    parts.append("![]({})".format(figures["sampling_interval"]))
    parts.append("")
    parts.append("### 2.2 油量噪声")
    parts.append("")
    parts.append(
        "停车样本单步 Δ油量标准差 σ = **{:.2f} L**，绝对值 95 分位 {:.2f} L。"
        "若把所有负向差分当作耗油，停车段合计 {:.1f} L，行驶段 {:.1f} L——"
        "噪声会淹没真实消耗，故后续不用该口径作为主结论。".format(
            quality_noise,
            sensor.get("stop_doil_p95_abs") or 0,
            sensor.get("naive_neg_sum_stop_l") or 0,
            sensor.get("naive_neg_sum_moving_l") or 0,
        )
    )
    parts.append("")
    sat = sensor.get("saturation") or {}
    parts.append(
        "油量读数达到 {:.0f} L 的样本 {} 条（占比 {:.1%}），加油量在加满时可能被截断低估。".format(
            sat.get("full_tank_l") or 320,
            sat.get("n_at_full") or 0,
            sat.get("ratio") or 0,
        )
    )
    parts.append("")
    parts.append("![]({})".format(figures["oil_timeseries"]))
    parts.append("")
    parts.append("![]({})".format(figures["noise_hist"]))
    parts.append("")
    gnss = sensor.get("gnss_offset_m") or {}
    if gnss:
        parts.append("### 2.3 GPS 与北斗偏差")
        parts.append("")
        parts.append(
            "北斗相对 GPS 的平面距离中位数 **{:.0f} m**，标准差仅 {:.0f} m（范围 {:.0f}–{:.0f} m）。"
            "这是稳定的系统偏差（坐标系或天线），不是随机定位抖动。"
            "若以后要把加油点对齐到真实加油站，必须先处理该偏移，否则会系统性对错。".format(
                gnss.get("median_m") or 0,
                gnss.get("std_m") or 0,
                gnss.get("min_m") or 0,
                gnss.get("max_m") or 0,
            )
        )
        parts.append("")
        if sensor.get("ad_oil_corr") is not None:
            parts.append(
                "`ADValue` 与 `oilValue` 的相关系数为 {:.2f}。"
                "真加油时期望油量上升、AD 下降；AD 突然打到 1500 以上视为传感器尖峰。".format(
                    float(sensor["ad_oil_corr"])
                )
            )
            parts.append("")


    parts.append("## 3. 加油识别与作业四项结果")
    parts.append("")
    parts.append(
        "检测规则：速度 ≤ 5 km/h、位置位移小、油量连续上升；间隔小于 3 分钟的上升合并为一次。"
        "单次加油量阈值 = max(25 L, 4σ) = **{:.1f} L**。随后用 `ADValue` 交叉验证并分层；"
        "长停车中不能被已有事件解释的净上升记为慢加油。".format(threshold_l)
    )
    parts.append("")
    conf = hw.get("confidence") or {}
    high = hw.get("high_confidence_consumption") or {}
    usable = hw.get("usable_consumption") or {}
    parts.append("| 指标 | 全部事件 | 高可信+较可信 |")
    parts.append("| --- | --- | --- |")
    parts.append("| 加油次数 | {} | {} |".format(hw.get("n_refuel"), usable.get("n_refuel", "")))
    parts.append("| 总加油量 (L) | {} | {} |".format(hw.get("total_refuel_l"), usable.get("total_refuel_l", "")))
    parts.append("| 耗油量（平衡，L） | {} | {} |".format(hw.get("consumption_balanced_l"), usable.get("consumption_balanced_l", "")))
    parts.append("| 对照：负向差分求和 (L) | {} | — |".format(hw.get("consumption_naive_neg_sum_l")))
    parts.append("| 期初 / 期末油量 (L) | {} / {} | 同左 |".format(hw.get("oil_first"), hw.get("oil_last")))
    parts.append("| 其中慢加油回补 | {} | — |".format(hw.get("n_slow_refuel", 0)))
    parts.append("| 仅高可信次数 / 加油量 | {} 次 / {} L | — |".format(high.get("n_refuel", 0), high.get("total_refuel_l", "")))
    parts.append("")
    parts.append(
        "可信度：高可信 {} 次 / {:.1f} L，较可信 {} 次 / {:.1f} L，存疑 {} 次 / {:.1f} L。"
        "管理核算以「高可信+较可信」为准；存疑事件只列入核查。"
        "仅高可信样本过少时，不宜单独做油量平衡。".format(
            (conf.get("高可信") or {}).get("n", 0),
            (conf.get("高可信") or {}).get("volume_l", 0),
            (conf.get("较可信") or {}).get("n", 0),
            (conf.get("较可信") or {}).get("volume_l", 0),
            (conf.get("存疑") or {}).get("n", 0),
            (conf.get("存疑") or {}).get("volume_l", 0),
        )
    )
    parts.append("")
    parts.append("平衡式：耗油 = 期初油量 + 总加油量 − 期末油量。对照口径显著偏大，说明传感器抖动不可直接积分。")
    parts.append("")
    parts.append("![]({})".format(figures["oil_refuel"]))
    parts.append("")
    parts.append("### 3.1 加油事件明细")
    parts.append("")
    parts.append(_events_table(events))
    parts.append("")

    parts.append("## 4. 运营与油耗管理挖掘")
    parts.append("")
    parts.append("### 4.1 按日运营台账")
    parts.append("")
    parts.append(_daily_table(ops.get("daily") or []))
    parts.append("")
    parts.append("![]({})".format(figures["daily_km"]))
    parts.append("")
    parts.append("### 4.2 轨迹与加油地点")
    parts.append("")
    parts.append("下图为 GPS 散点（颜色表示当时油量）与加油点。第一版不叠底图。")
    parts.append("")
    parts.append("![]({})".format(figures["trajectory"]))
    parts.append("")
    parts.append("### 4.3 长停车")
    parts.append("")
    parts.append("持续 ≥ 30 分钟的停车段如下（用于区分在途行驶与驻车/修整）。")
    parts.append("")
    parts.append(_stops_table(ops.get("long_stops") or []))
    parts.append("")
    parts.append("### 4.4 短时陡降（偷油/漏油候选）")
    parts.append("")
    parts.append("与「长停车整天净变化」不同：盗油通常是停车后数分钟内陡降，且短时间不回升。AD 应与油量反向（油降 AD 升）。")
    parts.append("")
    steep = ops.get("steep_drops") or []
    if not steep:
        parts.append("未发现满足窗口与回升过滤的短时陡降。")
    else:
        parts.append(_steep_table(steep))
    parts.append("")
    parts.append("### 4.5 长停车净变化（对照，不作盗油主证据）")
    parts.append("")
    anomalies = ops.get("parking_anomalies") or []
    if not anomalies:
        parts.append("长停车净下降均未超过噪声带。")
    else:
        parts.append(_anomaly_table(anomalies))
    parts.append("")
    parts.append("### 4.6 加油时机")
    parts.append("")
    timing = ops.get("refuel_timing") or {}
    if timing.get("n"):
        parts.append(
            "夜间（22:00–06:00）加油 {} / {} 次（占比 {:.0%}）。"
            "加前油量中位数 {:.1f} L；油箱低于约 30%（≤96 L）才加的 {} 次；"
            "油位已高于 200 L 仍小额加注的 {} 次（后一类更像误检或顶油）。".format(
                timing.get("night_n") or 0,
                timing.get("n") or 0,
                timing.get("night_ratio") or 0,
                timing.get("oil_start_median") or 0,
                timing.get("low_tank_n") or 0,
                timing.get("high_tank_topup_n") or 0,
            )
        )
        parts.append("")
        if timing.get("low_tank_events"):
            parts.append("低油位加油：")
            parts.append("")
            parts.append(_timing_table(timing["low_tank_events"]))
            parts.append("")
        if timing.get("high_tank_topups"):
            parts.append("高油位小额加注（待核）：")
            parts.append("")
            parts.append(_timing_table(timing["high_tank_topups"]))
            parts.append("")
    else:
        parts.append("无加油事件，无法统计时机。")
        parts.append("")
    parts.append("### 4.7 加油间隔段油耗")
    parts.append("")
    parts.append("仅用非「存疑」加油作为分段锚点，且里程 > 50 km 时估算百公里油耗；**不把 30 秒差分当作瞬时油耗**。")
    parts.append("")
    parts.append(_economy_table(ops.get("inter_refuel_economy") or []))
    parts.append("")

    parts.append("## 5. 给管理人员的建议")
    parts.append("")
    parts.append(_render_advice(advice))
    parts.append("")

    parts.append("## 6. 方法局限")
    parts.append("")
    parts.append("- AD 尖峰与油量小跳会制造假加油，必须以交叉验证分层，不能把 12 次都当确定加油。")
    parts.append("- 停车时油量仍有约 ±10 L 量级抖动，阈值过低会把噪声当成加油。")
    parts.append("- 读数顶死在 320 L 时，加满油量可能被低估。")
    parts.append("- 瞬时百公里油耗被噪声主导，本报告不将其作为主结论。")
    parts.append("- 数据为单车约 9 天，不能推广为司机画像或下周需求预测。")
    parts.append("- GPS/北斗有约 1.3 km 系统偏差，加油地点未做地图匹配。")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("报告由 `src.fuel_mgmt.pipeline` 生成。")
    parts.append("")
    return "\n".join(parts)


def write_report(text: str, path: Path) -> None:
    """写入 UTF-8 Markdown 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_advice(advice: str | None) -> str:
    """第 5 章正文：成功 / 未配置 Key / 调用失败。"""
    if advice is None:
        return (
            "未生成。未检测到 `DEEPSEEK_API_KEY` 或作业根目录的 `deepseek-api.txt`。"
            "配置 DeepSeek 密钥后重新运行流水线即可写入本章。"
            "前四章统计结果不受影响。"
        )
    if advice.startswith("__LLM_ERROR__:"):
        msg = advice.split(":", 1)[-1]
        return (
            "未生成。调用 DeepSeek 时出错：`{}`。"
            "请检查网络、额度与 `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL`。前四章不受影响。"
        ).format(msg)
    return advice


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """简单 Markdown 表。"""
    if not rows:
        return "（无记录）"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(lines)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).replace("|", "\\|")
    return s


def _events_table(events: list[RefuelEvent]) -> str:
    rows = []
    for i, e in enumerate(events, 1):
        ad = ""
        if e.ad_start is not None and e.ad_end is not None:
            ad = "{:.0f}→{:.0f}".format(e.ad_start, e.ad_end)
        src = "慢加" if e.source == "slow_stop" else "快加"
        rows.append(
            [
                i,
                e.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                e.end_time.strftime("%H:%M:%S"),
                "{:.4f}, {:.4f}".format(e.lat, e.lng),
                round(e.oil_start, 1),
                round(e.oil_end, 1),
                round(e.volume_l, 1),
                "是" if e.is_full else "否",
                e.confidence or "",
                ad,
                src,
            ]
        )
    return _md_table(
        ["序号", "开始时间", "结束", "位置", "油量前", "油量后", "加油量 L", "加满", "可信度", "AD", "来源"],
        rows,
    )


def _daily_table(daily: list[dict[str, Any]]) -> str:
    rows = [
        [
            r["date"],
            r["km"],
            r["oil_start"],
            r["oil_end"],
            r["stop_ratio"],
            r["n_refuel"],
            r["refuel_l"],
        ]
        for r in daily
    ]
    return _md_table(
        ["日期", "里程 km", "日初油量", "日末油量", "停车占比", "加油次数", "加油量 L"],
        rows,
    )


def _stops_table(stops: list[dict[str, Any]]) -> str:
    rows = [
        [s["t0"], s["t1"], s["dur_min"], "{:.4f}, {:.4f}".format(s["lat"], s["lng"]), s["oil_start"], s["oil_end"], s["doil"]]
        for s in stops
    ]
    return _md_table(
        ["开始", "结束", "时长 min", "位置", "油量前", "油量后", "净变化 L"],
        rows,
    )


def _anomaly_table(anomalies: list[dict[str, Any]]) -> str:
    rows = [
        [a["t0"], a["t1"], a["dur_min"], a["drop_l"], a.get("threshold_l", ""), "{:.4f}, {:.4f}".format(a["lat"], a["lng"])]
        for a in anomalies
    ]
    return _md_table(
        ["开始", "结束", "时长 min", "下降 L", "阈值 L", "位置"],
        rows,
    )


def _steep_table(rows_data: list[dict[str, Any]]) -> str:
    rows = [
        [
            a["t0"],
            a["t1"],
            a["dur_min"],
            a["drop_l"],
            a.get("threshold_l", ""),
            "{:.4f}, {:.4f}".format(a["lat"], a["lng"]),
            a.get("note", ""),
        ]
        for a in rows_data
    ]
    return _md_table(
        ["开始", "结束", "时长 min", "下降 L", "阈值 L", "位置", "AD 说明"],
        rows,
    )


def _timing_table(items: list[dict[str, Any]]) -> str:
    rows = [[r["time"], r["oil_start"], r["volume_l"], r.get("confidence", "")] for r in items]
    return _md_table(["时间", "加前油量 L", "加油量 L", "可信度"], rows)


def _economy_table(segs: list[dict[str, Any]]) -> str:
    rows = [
        [s["from"], s["to"], s["km"], s["used_l"], s.get("l_per_100km") if s.get("l_per_100km") is not None else "—", s.get("note", "")]
        for s in segs
    ]
    return _md_table(
        ["从", "到", "里程 km", "耗油 L", "L/100km", "说明"],
        rows,
    )
