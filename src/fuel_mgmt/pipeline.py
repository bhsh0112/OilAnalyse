"""一键生成油耗分析报告的命令行入口。

用法::

    python -m src.fuel_mgmt.pipeline --data "data/xxx.xls" --out outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .evidence import (
    confidence_counts,
    detect_slow_refuels,
    detect_steep_drops,
    merge_refuel_events,
    refuel_timing_stats,
    score_refuel_events,
)
from .load import load_trace
from .llm_advisor import generate_advice
from .mining import daily_ledger, inter_refuel_economy, long_stops, parking_anomalies
from .quality import analyze_quality
from .refuel import compute_consumption, detect_refuel_events, refuel_volume_threshold
from .report import render_report, write_report
from .summary import build_summary, write_summary
from .visualize import generate_figures


def project_root() -> Path:
    """作业根目录（src/fuel_mgmt 的上两级）。"""
    return Path(__file__).resolve().parents[2]


def run(data: Path, out_dir: Path, skip_llm: bool = False) -> dict[str, Path]:
    """执行完整分析并写出报告、图片与 JSON。

    Args:
        data: xls 路径。
        out_dir: 输出目录。
        skip_llm: 为 True 时不调用 DeepSeek。

    Returns:
        主要产物路径。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    df = load_trace(data)
    quality = analyze_quality(df)
    sigma = float(quality["noise"]["sigma_l"])
    threshold = refuel_volume_threshold(sigma)
    fast_events = detect_refuel_events(df, sigma)
    stops = long_stops(df)
    slow_events = detect_slow_refuels(df, fast_events, stops, sigma)
    events = merge_refuel_events(fast_events, slow_events)
    events = score_refuel_events(df, events)
    consumption = compute_consumption(df, events)
    high_events = [e for e in events if e.confidence == "高可信"]
    usable_events = [e for e in events if e.confidence in ("高可信", "较可信")]
    consumption_high = compute_consumption(df, high_events)
    consumption_usable = compute_consumption(df, usable_events)
    daily = daily_ledger(df, events)
    anomalies = parking_anomalies(stops, events, sigma)
    steep_drops = detect_steep_drops(df, events, sigma)
    economy = inter_refuel_economy(df, usable_events or events)
    timing = refuel_timing_stats(events)
    conf_counts = confidence_counts(events)

    figures = generate_figures(df, events, daily, fig_dir)
    summary = build_summary(
        data_path=str(data),
        quality=quality,
        events=events,
        consumption=consumption,
        daily=daily,
        stops=stops,
        anomalies=anomalies,
        economy=economy,
        threshold_l=threshold,
        consumption_high=consumption_high,
        consumption_usable=consumption_usable,
        steep_drops=steep_drops,
        timing=timing,
        confidence=conf_counts,
        n_slow=sum(1 for e in events if e.source == "slow_stop"),
    )
    summary_path = out_dir / "analysis_summary.json"
    write_summary(summary, summary_path)

    advice = None if skip_llm else generate_advice(summary)
    report_text = render_report(
        summary=summary,
        events=events,
        figures=figures,
        advice=advice,
        threshold_l=threshold,
    )
    report_path = out_dir / "report.md"
    write_report(report_text, report_path)

    return {
        "report": report_path,
        "summary": summary_path,
        "figures": fig_dir,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    load_dotenv(project_root() / ".env")
    load_dotenv()

    parser = argparse.ArgumentParser(description="营运车辆油耗分析报告")
    parser.add_argument("--data", required=True, help="xls 数据路径")
    parser.add_argument("--out", default="outputs", help="输出目录")
    parser.add_argument("--skip-llm", action="store_true", help="不调用 DeepSeek")
    args = parser.parse_args(argv)

    data = Path(args.data)
    if not data.is_absolute():
        cand = Path.cwd() / data
        data = cand if cand.exists() else project_root() / data
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir

    artifacts = run(data, out_dir, skip_llm=bool(args.skip_llm))
    print("报告:", artifacts["report"])
    print("摘要:", artifacts["summary"])
    print("图片:", artifacts["figures"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
