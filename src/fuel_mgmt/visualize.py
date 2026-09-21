"""绘图：少而清楚，Windows 下优先使用微软雅黑。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .config import FULL_TANK_L
from .refuel import RefuelEvent


def setup_plot_style() -> None:
    """设置中文字体与负号，避免 Windows 下图中出现方块字。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 140
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25


def generate_figures(
    df: pd.DataFrame,
    events: list[RefuelEvent],
    daily: list[dict[str, Any]],
    fig_dir: Path,
) -> dict[str, str]:
    """生成报告所需全部图片，返回 {逻辑名: 相对路径}。

    Args:
        df: 轨迹表。
        events: 加油事件。
        daily: 日台账。
        fig_dir: 图片输出目录。

    Returns:
        Markdown 可用的相对路径（相对于 report.md）。
    """
    setup_plot_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "oil_timeseries": fig_dir / "oil_timeseries.png",
        "oil_refuel": fig_dir / "oil_refuel.png",
        "noise_hist": fig_dir / "noise_hist.png",
        "sampling_interval": fig_dir / "sampling_interval.png",
        "daily_km": fig_dir / "daily_km.png",
        "trajectory": fig_dir / "trajectory.png",
    }
    plot_oil_timeseries(df, paths["oil_timeseries"])
    plot_oil_refuel(df, events, paths["oil_refuel"])
    plot_noise_hist(df, paths["noise_hist"])
    plot_sampling_interval(df, paths["sampling_interval"])
    plot_daily_km(daily, paths["daily_km"])
    plot_trajectory(df, events, paths["trajectory"])
    return {k: f"figures/{p.name}" for k, p in paths.items()}


def plot_oil_timeseries(df: pd.DataFrame, out: Path) -> None:
    """油量–时间全曲线，并标出 320 L 饱和线。"""
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(df["GPSTime"], df["oilValue"], color="#1f4e79", lw=1.0, label="油量")
    ax.axhline(FULL_TANK_L, color="#c0392b", ls="--", lw=1.0, label=f"饱和 {FULL_TANK_L:.0f} L")
    ax.set_ylabel("油量 (L)")
    ax.set_xlabel("GPSTime")
    ax.set_title("油量时间序列")
    ax.legend(loc="upper right", frameon=True)
    _format_time_axis(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_oil_refuel(df: pd.DataFrame, events: Iterable[RefuelEvent], out: Path) -> None:
    """油量曲线上按可信度着色标注加油区间。"""
    events = list(events)
    colors = {"高可信": "#27ae60", "较可信": "#e67e22", "存疑": "#c0392b", "": "#7f8c8d"}
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(df["GPSTime"], df["oilValue"], color="#1f4e79", lw=1.0, label="油量")
    ax.axhline(FULL_TANK_L, color="#c0392b", ls="--", lw=1.0, label=f"饱和 {FULL_TANK_L:.0f} L")
    seen: set[str] = set()
    for e in events:
        color = colors.get(e.confidence, "#7f8c8d")
        label = e.confidence or "加油"
        ax.axvspan(e.start_time, e.end_time, color=color, alpha=0.22, label=label if label not in seen else None)
        seen.add(label)
        ax.scatter([e.start_time], [e.oil_start], color=color, s=28, zorder=3)
    ax.set_ylabel("油量 (L)")
    ax.set_xlabel("GPSTime")
    ax.set_title("加油事件（绿=高可信，橙=较可信，红=存疑）")
    ax.legend(loc="upper right", frameon=True)
    _format_time_axis(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_noise_hist(df: pd.DataFrame, out: Path) -> None:
    """停车 vs 行驶的单步油量变化分布。"""
    stop = df.loc[df["is_stop"] & df["doil"].notna(), "doil"]
    move = df.loc[(~df["is_stop"]) & df["doil"].notna(), "doil"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bins = range(-30, 31, 1)
    ax.hist(stop, bins=bins, alpha=0.65, label=f"停车 n={len(stop)}", color="#7f8c8d")
    ax.hist(move, bins=bins, alpha=0.55, label=f"行驶 n={len(move)}", color="#2980b9")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-30, 30)
    ax.set_xlabel("单步 Δ油量 (L)")
    ax.set_ylabel("频数")
    ax.set_title("油量噪声：停车 vs 行驶")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_sampling_interval(df: pd.DataFrame, out: Path) -> None:
    """采样间隔分布（截断长尾以便观察主峰）。"""
    dt = df["dt_s"].dropna()
    dt_clip = dt[dt <= 300]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(dt_clip, bins=40, color="#34495e", alpha=0.85)
    if len(dt):
        ax.axvline(dt.median(), color="#e67e22", ls="--", label=f"中位数 {dt.median():.0f} s")
    ax.set_xlabel("采样间隔 (s，已截断至 300 s)")
    ax.set_ylabel("频数")
    ax.set_title("GPSTime 采样间隔")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_daily_km(daily: list[dict[str, Any]], out: Path) -> None:
    """按日行驶里程柱状图。"""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    if daily:
        xs = [r["date"][5:] for r in daily]  # MM-DD
        ys = [r["km"] for r in daily]
        ax.bar(xs, ys, color="#1f4e79", width=0.6)
    ax.set_ylabel("行驶里程 (km)")
    ax.set_xlabel("日期")
    ax.set_title("按日行驶里程")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_trajectory(df: pd.DataFrame, events: Iterable[RefuelEvent], out: Path) -> None:
    """经纬度散点轨迹，叠加加油点。"""
    events = list(events)
    fig, ax = plt.subplots(figsize=(7.5, 7.2))
    ax.scatter(
        df["GPSlng"],
        df["GPSlat"],
        s=4,
        c=pd.to_numeric(df["oilValue"], errors="coerce"),
        cmap="viridis",
        alpha=0.35,
        linewidths=0,
        label="轨迹",
    )
    if events:
        ax.scatter(
            [e.lng for e in events],
            [e.lat for e in events],
            s=70,
            c="#c0392b",
            marker="*",
            zorder=4,
            label="加油点",
        )
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title("行驶轨迹与加油点")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _format_time_axis(ax: plt.Axes) -> None:
    """压缩时间轴刻度，避免挤成一团。"""
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig = ax.get_figure()
    if fig is not None:
        fig.autofmt_xdate(rotation=0, ha="center")
