"""加油事件检测与油量平衡耗油核算。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    FULL_MARK_L,
    OIL_RISE_EPS_L,
    REFUEL_MAX_DISPLACEMENT_M,
    REFUEL_MAX_SPEED_KMH,
    REFUEL_MERGE_GAP_S,
    REFUEL_MIN_VOLUME_L,
    REFUEL_SIGMA_MULT,
)


@dataclass
class RefuelEvent:
    """一次合并后的加油事件。"""

    start_time: datetime
    end_time: datetime
    lat: float
    lng: float
    oil_start: float
    oil_end: float
    volume_l: float
    is_full: bool
    n_samples: int
    start_idx: int
    end_idx: int
    source: str = "fast"
    confidence: str = ""
    ad_ok: bool | None = None
    ad_start: float | None = None
    ad_end: float | None = None
    ad_delta: float | None = None
    ad_spike: bool = False
    reasons: list[str] = field(default_factory=list)
    confidence_ad: str = ""
    station_name: str | None = None
    station_distance_m: float | None = None
    station_band: str = ""
    station_source: str = ""
    station_query_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        d = asdict(self)
        d["start_time"] = _fmt(self.start_time)
        d["end_time"] = _fmt(self.end_time)
        d["volume_l"] = round(float(self.volume_l), 1)
        d["oil_start"] = round(float(self.oil_start), 1)
        d["oil_end"] = round(float(self.oil_end), 1)
        d["lat"] = round(float(self.lat), 6)
        d["lng"] = round(float(self.lng), 6)
        for key in ("ad_start", "ad_end", "ad_delta"):
            if d[key] is not None:
                d[key] = round(float(d[key]), 1)
        if d["station_distance_m"] is not None:
            d["station_distance_m"] = round(float(d["station_distance_m"]), 1)
        return d


def refuel_volume_threshold(sigma: float) -> float:
    """由停车噪声确定最小加油量阈值。"""
    return max(REFUEL_MIN_VOLUME_L, REFUEL_SIGMA_MULT * float(sigma))


def detect_refuel_events(df: pd.DataFrame, sigma: float) -> list[RefuelEvent]:
    """检测并合并加油事件。

    规则：低速、位置几乎不动、油量连续上升；间隔小于 3 分钟的上升段合并。
    加油量 = 结束油量 − 开始油量（开始油量取上升前一条记录）。

    Args:
        df: 已按 GPSTime 排序并派生 doil 的轨迹。
        sigma: 停车油量噪声标准差（升）。

    Returns:
        通过体积阈值过滤后的加油事件列表。
    """
    min_vol = refuel_volume_threshold(sigma)
    events: list[RefuelEvent] = []
    cur: dict[str, Any] | None = None

    def close_event() -> None:
        nonlocal cur
        if cur is None:
            return
        i0 = int(cur["start_idx"])
        i1 = int(cur["last_idx"])
        oil_end = float(df.at[i1, "oilValue"])
        volume = oil_end - float(cur["oil_start"])
        if volume >= min_vol:
            events.append(
                RefuelEvent(
                    start_time=pd.Timestamp(df.at[i0, "GPSTime"]).to_pydatetime(),
                    end_time=pd.Timestamp(df.at[i1, "GPSTime"]).to_pydatetime(),
                    lat=float(df.at[i0, "GPSlat"]),
                    lng=float(df.at[i0, "GPSlng"]),
                    oil_start=float(cur["oil_start"]),
                    oil_end=oil_end,
                    volume_l=float(volume),
                    is_full=oil_end >= FULL_MARK_L,
                    n_samples=int(i1 - i0 + 1),
                    start_idx=i0,
                    end_idx=i1,
                )
            )
        cur = None

    for i in range(len(df)):
        doil = df.at[i, "doil"]
        if pd.isna(doil):
            continue
        t = pd.Timestamp(df.at[i, "GPSTime"])
        spd = float(df.at[i, "GPSpdValue"]) if pd.notna(df.at[i, "GPSpdValue"]) else 0.0
        rising = (float(doil) > OIL_RISE_EPS_L) and (spd <= REFUEL_MAX_SPEED_KMH)

        if not rising:
            if cur is not None:
                last_t = pd.Timestamp(df.at[cur["last_idx"], "GPSTime"])
                if (t - last_t).total_seconds() > REFUEL_MERGE_GAP_S:
                    close_event()
            continue

        lat = float(df.at[i, "GPSlat"])
        lng = float(df.at[i, "GPSlng"])

        if cur is None:
            cur = _new_cluster(df, i, lat, lng)
            continue

        last_t = pd.Timestamp(df.at[cur["last_idx"], "GPSTime"])
        gap = (t - last_t).total_seconds()
        disp = haversine_m(cur["lat0"], cur["lng0"], lat, lng)
        if gap <= REFUEL_MERGE_GAP_S and disp <= REFUEL_MAX_DISPLACEMENT_M:
            cur["last_idx"] = i
        else:
            close_event()
            cur = _new_cluster(df, i, lat, lng)

    close_event()
    return events


def compute_consumption(df: pd.DataFrame, events: list[RefuelEvent]) -> dict[str, Any]:
    """用油量平衡计算耗油，并给出负向差分求和对照。

    平衡式：耗油 = 期初油量 + 总加油量 − 期末油量。

    Args:
        df: 轨迹表。
        events: 已识别的加油事件。

    Returns:
        含平衡耗油与朴素负差分对照的字典。
    """
    if df.empty:
        return {
            "oil_first": 0.0,
            "oil_last": 0.0,
            "n_refuel": 0,
            "total_refuel_l": 0.0,
            "consumption_balanced_l": 0.0,
            "consumption_naive_neg_sum_l": 0.0,
        }

    oil_first = float(df["oilValue"].iloc[0])
    oil_last = float(df["oilValue"].iloc[-1])
    total_refuel = float(sum(e.volume_l for e in events))
    balanced = oil_first + total_refuel - oil_last
    naive = float(-df.loc[df["doil"] < 0, "doil"].sum())
    return {
        "oil_first": round(oil_first, 1),
        "oil_last": round(oil_last, 1),
        "n_refuel": int(len(events)),
        "total_refuel_l": round(total_refuel, 1),
        "consumption_balanced_l": round(balanced, 1),
        "consumption_naive_neg_sum_l": round(naive, 1),
    }


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点球面距离，单位米。"""
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def _new_cluster(df: pd.DataFrame, i: int, lat: float, lng: float) -> dict[str, Any]:
    """以第 i 条上升记录开启一个加油簇。"""
    if i > 0:
        oil_start = float(df.at[i - 1, "oilValue"])
    else:
        oil_start = float(df.at[i, "oilValue"]) - float(df.at[i, "doil"])
    return {
        "start_idx": i,
        "last_idx": i,
        "oil_start": oil_start,
        "lat0": lat,
        "lng0": lng,
    }


def _fmt(ts: datetime) -> str:
    """格式化时间为 'YYYY-MM-DD HH:MM:SS'。"""
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
