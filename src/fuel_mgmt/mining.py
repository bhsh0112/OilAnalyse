"""管理向挖掘：日台账、长停车、驻车异常、加油间隔油耗。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from .config import (
    LONG_STOP_MIN_S,
    MIN_SEGMENT_KM,
    PARKING_DROP_L,
    PARKING_SIGMA_MULT,
    STOP_SPEED_KMH,
)
from .refuel import RefuelEvent


def daily_ledger(df: pd.DataFrame, events: list[RefuelEvent]) -> list[dict[str, Any]]:
    """按自然日汇总里程、油量与加油次数。

    Args:
        df: 轨迹表。
        events: 加油事件。

    Returns:
        按日期排序的日台账列表。
    """
    if df.empty:
        return []

    refuel_by_day: dict[Any, list[RefuelEvent]] = {}
    for e in events:
        day = pd.Timestamp(e.start_time).date()
        refuel_by_day.setdefault(day, []).append(e)

    rows: list[dict[str, Any]] = []
    for day, g in df.groupby("date", sort=True):
        km = float(g["mileageValue"].iloc[-1] - g["mileageValue"].iloc[0]) * 0.1
        day_events = refuel_by_day.get(day, [])
        rows.append(
            {
                "date": str(day),
                "n_records": int(len(g)),
                "km": round(km, 1),
                "oil_start": round(float(g["oilValue"].iloc[0]), 1),
                "oil_end": round(float(g["oilValue"].iloc[-1]), 1),
                "oil_min": round(float(g["oilValue"].min()), 1),
                "oil_max": round(float(g["oilValue"].max()), 1),
                "stop_ratio": round(float(g["is_stop"].mean()), 3),
                "spd_mean": round(float(g["GPSpdValue"].mean()), 2),
                "n_refuel": int(len(day_events)),
                "refuel_l": round(float(sum(e.volume_l for e in day_events)), 1),
            }
        )
    return rows


def long_stops(df: pd.DataFrame, min_duration_s: float = LONG_STOP_MIN_S) -> list[dict[str, Any]]:
    """提取持续超过阈值的停车段。

    Args:
        df: 轨迹表。
        min_duration_s: 最短停车时长（秒）。

    Returns:
        长停车列表。
    """
    if df.empty:
        return []

    is_stop = df["GPSpdValue"].fillna(0) <= STOP_SPEED_KMH
    grp = (is_stop != is_stop.shift()).cumsum()
    out: list[dict[str, Any]] = []
    for _, g in df.groupby(grp):
        if not bool(g["GPSpdValue"].fillna(0).le(STOP_SPEED_KMH).all()):
            continue
        t0 = pd.Timestamp(g["GPSTime"].iloc[0])
        t1 = pd.Timestamp(g["GPSTime"].iloc[-1])
        dur = (t1 - t0).total_seconds()
        if dur < min_duration_s:
            continue
        i0 = int(g.index[0])
        i1 = int(g.index[-1])
        oil0 = float(g["oilValue"].iloc[0])
        oil1 = float(g["oilValue"].iloc[-1])
        out.append(
            {
                "t0": t0.strftime("%Y-%m-%d %H:%M:%S"),
                "t1": t1.strftime("%Y-%m-%d %H:%M:%S"),
                "dur_min": round(dur / 60.0, 1),
                "lat": round(float(g["GPSlat"].median()), 6),
                "lng": round(float(g["GPSlng"].median()), 6),
                "oil_start": round(oil0, 1),
                "oil_end": round(oil1, 1),
                "doil": round(oil1 - oil0, 1),
                "n_samples": int(len(g)),
                "start_idx": i0,
                "end_idx": i1,
                "t0_ts": t0,
                "t1_ts": t1,
            }
        )
    return out


def parking_anomalies(
    stops: list[dict[str, Any]],
    events: list[RefuelEvent],
    sigma: float,
) -> list[dict[str, Any]]:
    """长停车期间油量净下降超过噪声带的记录，供漏油/盗油核查。

    与加油事件时间重叠的停车段不列入（加油表现为上升）。

    Args:
        stops: `long_stops` 输出。
        events: 加油事件。
        sigma: 停车噪声标准差。

    Returns:
        异常停车列表（已去掉内部时间戳）。
    """
    thresh = max(PARKING_DROP_L, PARKING_SIGMA_MULT * float(sigma))
    anomalies: list[dict[str, Any]] = []
    for s in stops:
        t0: datetime = s["t0_ts"]
        t1: datetime = s["t1_ts"]
        if _overlaps_refuel(t0, t1, events):
            continue
        drop = -float(s["doil"])
        if drop <= thresh:
            continue
        item = {k: v for k, v in s.items() if k not in ("t0_ts", "t1_ts")}
        item["drop_l"] = round(drop, 1)
        item["threshold_l"] = round(thresh, 1)
        anomalies.append(item)
    return anomalies


def inter_refuel_economy(
    df: pd.DataFrame,
    events: list[RefuelEvent],
) -> list[dict[str, Any]]:
    """两次加油之间（含数据起止）的分段油耗。

    仅在该段里程大于 50 km 时计算 L/100km，并标注不确定性。

    Args:
        df: 轨迹表。
        events: 按时间排序的加油事件。

    Returns:
        分段油耗列表。
    """
    if df.empty:
        return []

    ordered = sorted(events, key=lambda e: e.start_time)
    last_i = int(len(df) - 1)

    def _before_idx(event: RefuelEvent) -> int:
        return max(0, int(event.start_idx) - 1)

    # 段边界：数据起点 → 各次加油前 → 各次加油后 → 数据终点
    # 只在「加油后 → 下一次加油前」以及起止两端计算耗油。
    anchors: list[tuple[int, float, str]] = []
    anchors.append((0, float(df["oilValue"].iloc[0]), "数据起点"))
    for event in ordered:
        b = _before_idx(event)
        anchors.append((b, float(event.oil_start), f"加油前 {event.start_time.strftime('%m-%d %H:%M')}"))
        anchors.append(
            (int(event.end_idx), float(event.oil_end), f"加油后 {event.end_time.strftime('%m-%d %H:%M')}")
        )
    anchors.append((last_i, float(df["oilValue"].iloc[-1]), "数据终点"))

    segments: list[dict[str, Any]] = []
    for i in range(len(anchors) - 1):
        i0, oil0, label0 = anchors[i]
        i1, oil1, label1 = anchors[i + 1]
        # 跳过「加油前 → 加油后」这一段（那是加油，不是耗油）
        if label0.startswith("加油前") and label1.startswith("加油后"):
            continue
        if i1 <= i0:
            continue
        km = float(df["mileageValue"].iloc[i1] - df["mileageValue"].iloc[i0]) * 0.1
        used = oil0 - oil1
        item: dict[str, Any] = {
            "from": label0,
            "to": label1,
            "t0": pd.Timestamp(df["GPSTime"].iloc[i0]).strftime("%Y-%m-%d %H:%M:%S"),
            "t1": pd.Timestamp(df["GPSTime"].iloc[i1]).strftime("%Y-%m-%d %H:%M:%S"),
            "km": round(km, 1),
            "oil_start": round(oil0, 1),
            "oil_end": round(oil1, 1),
            "used_l": round(used, 1),
            "l_per_100km": None,
            "note": "",
        }
        if km < MIN_SEGMENT_KM:
            item["note"] = f"里程不足 {MIN_SEGMENT_KM:.0f} km，不计算百公里油耗"
        elif used <= 0:
            item["note"] = "该段油量未净下降（可能含未识别加油或饱和截断），不计算百公里油耗"
        else:
            item["l_per_100km"] = round(used / km * 100.0, 1)
            item["note"] = "由加油间隔油量差估算，受传感器噪声与 320 饱和影响，仅供对照"
        segments.append(item)
    return segments


def strip_stop_internal_ts(stops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """去掉长停车里仅供内部判断的 Timestamp 字段。"""
    cleaned = []
    for s in stops:
        cleaned.append({k: v for k, v in s.items() if k not in ("t0_ts", "t1_ts")})
    return cleaned


def _overlaps_refuel(t0: datetime, t1: datetime, events: list[RefuelEvent]) -> bool:
    """停车时段是否与任一加油事件相交。"""
    t0 = pd.Timestamp(t0)
    t1 = pd.Timestamp(t1)
    for e in events:
        es, ee = pd.Timestamp(e.start_time), pd.Timestamp(e.end_time)
        if es <= t1 and ee >= t0:
            return True
    return False
