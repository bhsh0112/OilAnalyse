"""加油可信度、AD 交叉验证、慢加油漏检与短时陡降。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    AD_DECREASE_MIN,
    AD_SPIKE_ABOVE_MEDIAN,
    AD_SPIKE_ABS,
    FULL_MARK_L,
    FULL_TANK_L,
    SLOW_REFUEL_RESIDUAL_L,
    STEEP_DROP_L,
    STEEP_DROP_RECOVER_S,
    STEEP_DROP_SIGMA_MULT,
    STEEP_DROP_WINDOW_S,
    STOP_SPEED_KMH,
)
from .refuel import RefuelEvent, refuel_volume_threshold


def score_refuel_events(df: pd.DataFrame, events: list[RefuelEvent]) -> list[RefuelEvent]:
    """用 AD 通道与事件形态给加油事件分层。

    真加油常见：油量平滑上升且 ADValue 同步下降。
    假事件常见：单点跳变、AD 尖峰、加油量接近噪声。

    Args:
        df: 轨迹表。
        events: 未评分或已评分的加油事件。

    Returns:
        写入 confidence / ad_* / reasons 后的同一批事件。
    """
    scored: list[RefuelEvent] = []
    for event in events:
        ad = _ad_features(df, event)
        reasons = list(ad["reasons"])
        volume = float(event.volume_l)
        n_samples = int(event.n_samples)
        ad_ok = ad["ad_ok"]
        ad_spike = bool(ad["ad_spike"])

        if volume < 50:
            reasons.append("加油量低于 50 L，接近噪声或顶油")
        if n_samples <= 1:
            reasons.append("仅 1 个上升采样点")
        if event.source == "slow_stop":
            reasons.append("来自长停车净上升回补，非快速加油簇")

        if ad_spike or n_samples <= 1 or volume < 50:
            confidence = "存疑"
        elif ad_ok is False and ad.get("ad_delta") is not None and float(ad["ad_delta"]) > 0:
            confidence = "存疑"
            reasons.append("油量上升但 AD 未下降，形态不支持真加油")
        elif volume >= 100 and ad_ok is True and n_samples >= 3:
            confidence = "高可信"
            reasons.append("油量连续大幅上升且 AD 同步下降")
        elif volume >= 80 and ad_ok is True:
            confidence = "较可信"
        elif volume >= 100 and n_samples >= 4 and not ad_spike:
            confidence = "较可信"
            reasons.append("形态像加油但 AD 证据不足")
        else:
            confidence = "存疑"

        event.confidence = confidence
        event.ad_ok = ad_ok
        event.ad_start = ad["ad_start"]
        event.ad_end = ad["ad_end"]
        event.ad_delta = ad["ad_delta"]
        event.ad_spike = ad_spike
        event.reasons = reasons
        scored.append(event)
    return scored


def detect_slow_refuels(
    df: pd.DataFrame,
    events: list[RefuelEvent],
    stops: list[dict[str, Any]],
    sigma: float,
) -> list[RefuelEvent]:
    """在长停车中回补「慢加油」：净上升不能被已有事件解释。

    Args:
        df: 轨迹表。
        events: 已识别的快速加油簇。
        stops: `long_stops` 输出（需含 start_idx/end_idx）。
        sigma: 停车噪声标准差。

    Returns:
        新增的慢加油事件（source=slow_stop）。
    """
    min_vol = max(SLOW_REFUEL_RESIDUAL_L, refuel_volume_threshold(sigma))
    extra: list[RefuelEvent] = []
    for stop in stops:
        if "start_idx" not in stop:
            continue
        i0 = int(stop["start_idx"])
        i1 = int(stop["end_idx"])
        t0 = pd.Timestamp(stop["t0_ts"])
        t1 = pd.Timestamp(stop["t1_ts"])
        explained = 0.0
        for event in events:
            es, ee = pd.Timestamp(event.start_time), pd.Timestamp(event.end_time)
            if es <= t1 and ee >= t0:
                explained += float(event.volume_l)
        residual = float(stop["oil_end"]) - float(stop["oil_start"]) - explained
        if residual < min_vol:
            continue

        oil_seg = df["oilValue"].iloc[i0 : i1 + 1]
        imin = int(oil_seg.idxmin())
        imax = int(oil_seg.idxmax())
        if imax <= imin:
            continue
        oil_start = float(df.at[imin, "oilValue"])
        oil_end = float(df.at[imax, "oilValue"])
        volume = oil_end - oil_start
        if volume < min_vol:
            continue

        ad_start = float(df.at[imin, "ADValue"]) if "ADValue" in df.columns else None
        ad_end = float(df.at[imax, "ADValue"]) if "ADValue" in df.columns else None
        if ad_start is not None and ad_end is not None and ad_end > ad_start - AD_DECREASE_MIN:
            # 油量升但 AD 不降，更像漂移
            continue

        extra.append(
            RefuelEvent(
                start_time=pd.Timestamp(df.at[imin, "GPSTime"]).to_pydatetime(),
                end_time=pd.Timestamp(df.at[imax, "GPSTime"]).to_pydatetime(),
                lat=float(df.at[imin, "GPSlat"]),
                lng=float(df.at[imin, "GPSlng"]),
                oil_start=oil_start,
                oil_end=oil_end,
                volume_l=volume,
                is_full=oil_end >= FULL_MARK_L,
                n_samples=int(imax - imin + 1),
                start_idx=imin,
                end_idx=imax,
                source="slow_stop",
            )
        )
    return extra


def merge_refuel_events(
    fast: list[RefuelEvent],
    slow: list[RefuelEvent],
) -> list[RefuelEvent]:
    """合并快/慢加油，去掉与已有事件时间重叠的慢加油。"""
    merged = list(fast)
    for event in slow:
        if _overlaps_any(event, merged):
            continue
        merged.append(event)
    merged.sort(key=lambda e: e.start_time)
    return merged


def detect_steep_drops(
    df: pd.DataFrame,
    events: list[RefuelEvent],
    sigma: float,
) -> list[dict[str, Any]]:
    """停车状态下短时大幅油量下降，供偷油/漏油核查。

    与长停车整天净变化不同：盗油通常是数分钟内陡降，且 5 分钟内不回升。

    Args:
        df: 轨迹表。
        events: 加油事件（用于排除加油邻域）。
        sigma: 停车噪声标准差。

    Returns:
        陡降事件列表。
    """
    if df.empty:
        return []
    thresh = max(STEEP_DROP_L, STEEP_DROP_SIGMA_MULT * float(sigma))
    out: list[dict[str, Any]] = []
    n = len(df)
    i = 0
    while i < n:
        if not _is_stop_row(df, i):
            i += 1
            continue
        t_i = pd.Timestamp(df.at[i, "GPSTime"])
        oil_i = float(df.at[i, "oilValue"])
        best: tuple[int, int, float] | None = None
        j = i + 1
        while j < n and _is_stop_row(df, j):
            t_j = pd.Timestamp(df.at[j, "GPSTime"])
            if (t_j - t_i).total_seconds() > STEEP_DROP_WINDOW_S:
                break
            drop = oil_i - float(df.at[j, "oilValue"])
            if drop >= thresh:
                best = (i, j, drop)
            j += 1
        if best is None:
            i += 1
            continue
        i0, i1, drop = best
        t0 = pd.Timestamp(df.at[i0, "GPSTime"])
        t1 = pd.Timestamp(df.at[i1, "GPSTime"])
        if _overlaps_any_times(t0, t1, events):
            i = i1 + 1
            continue
        if _recovers_soon(df, i1, float(df.at[i1, "oilValue"]) + drop * 0.7):
            i = i1 + 1
            continue
        ad_note = ""
        ad_ok = None
        if "ADValue" in df.columns:
            ad0 = float(df.at[i0, "ADValue"])
            ad1 = float(df.at[i1, "ADValue"])
            # 油量下降时 AD 应变大（与加油时 AD 下降相反）
            ad_ok = ad1 >= ad0
            ad_note = "AD 同步上升" if ad_ok else "AD 未同步，更像传感器跳变"
        out.append(
            {
                "t0": t0.strftime("%Y-%m-%d %H:%M:%S"),
                "t1": t1.strftime("%Y-%m-%d %H:%M:%S"),
                "dur_min": round((t1 - t0).total_seconds() / 60.0, 1),
                "lat": round(float(df.at[i0, "GPSlat"]), 6),
                "lng": round(float(df.at[i0, "GPSlng"]), 6),
                "oil_start": round(oil_i, 1),
                "oil_end": round(float(df.at[i1, "oilValue"]), 1),
                "drop_l": round(drop, 1),
                "threshold_l": round(thresh, 1),
                "ad_ok": ad_ok,
                "note": ad_note,
            }
        )
        i = i1 + 1
    return out


def refuel_timing_stats(events: list[RefuelEvent]) -> dict[str, Any]:
    """加油时机：夜间比例、加前油位、低油位冒险加满。"""
    if not events:
        return {
            "n": 0,
            "night_n": 0,
            "night_ratio": 0.0,
            "low_tank_n": 0,
            "high_tank_topup_n": 0,
            "oil_start_median": None,
        }
    oils = [float(e.oil_start) for e in events]
    night = [e for e in events if pd.Timestamp(e.start_time).hour < 6 or pd.Timestamp(e.start_time).hour >= 22]
    low = [e for e in events if e.oil_start <= 0.3 * FULL_TANK_L]
    topup = [e for e in events if e.oil_start >= 200 and e.volume_l < 80]
    return {
        "n": len(events),
        "night_n": len(night),
        "night_ratio": round(len(night) / len(events), 3),
        "low_tank_n": len(low),
        "high_tank_topup_n": len(topup),
        "oil_start_median": round(float(np.median(oils)), 1),
        "low_tank_events": [
            {
                "time": pd.Timestamp(e.start_time).strftime("%Y-%m-%d %H:%M:%S"),
                "oil_start": round(float(e.oil_start), 1),
                "volume_l": round(float(e.volume_l), 1),
                "confidence": e.confidence,
            }
            for e in low
        ],
        "high_tank_topups": [
            {
                "time": pd.Timestamp(e.start_time).strftime("%Y-%m-%d %H:%M:%S"),
                "oil_start": round(float(e.oil_start), 1),
                "volume_l": round(float(e.volume_l), 1),
                "confidence": e.confidence,
            }
            for e in topup
        ],
    }


def confidence_counts(events: list[RefuelEvent]) -> dict[str, Any]:
    """按可信度汇总次数与加油量。"""
    buckets = {"高可信": [], "较可信": [], "存疑": []}
    for event in events:
        buckets.setdefault(event.confidence or "存疑", []).append(event)
    out: dict[str, Any] = {}
    for name, items in buckets.items():
        out[name] = {
            "n": len(items),
            "volume_l": round(float(sum(e.volume_l for e in items)), 1),
        }
    return out


def _ad_features(df: pd.DataFrame, event: RefuelEvent) -> dict[str, Any]:
    """提取一次加油窗口的 AD 特征。"""
    reasons: list[str] = []
    if "ADValue" not in df.columns:
        return {
            "ad_ok": None,
            "ad_start": None,
            "ad_end": None,
            "ad_delta": None,
            "ad_spike": False,
            "reasons": reasons,
        }
    i0 = max(0, int(event.start_idx) - 1)
    i1 = int(event.end_idx)
    ads = df["ADValue"].iloc[i0 : i1 + 1].astype(float)
    ad_start = float(ads.iloc[0])
    ad_end = float(ads.iloc[-1])
    ad_delta = ad_end - ad_start
    median = float(ads.median())
    ad_max = float(ads.max())
    ad_spike = ad_max >= AD_SPIKE_ABS and (ad_max - median) >= AD_SPIKE_ABOVE_MEDIAN
    ad_ok = (ad_delta <= -AD_DECREASE_MIN) and (not ad_spike)
    if ad_spike:
        reasons.append(f"AD 尖峰 {ad_max:.0f}（中位 {median:.0f}）")
    if ad_delta <= -AD_DECREASE_MIN:
        reasons.append(f"AD {ad_start:.0f}→{ad_end:.0f}，与加油同向")
    elif ad_delta >= AD_DECREASE_MIN:
        reasons.append(f"AD {ad_start:.0f}→{ad_end:.0f}，与油量上升反向")
    return {
        "ad_ok": ad_ok,
        "ad_start": ad_start,
        "ad_end": ad_end,
        "ad_delta": ad_delta,
        "ad_spike": ad_spike,
        "reasons": reasons,
    }


def _is_stop_row(df: pd.DataFrame, i: int) -> bool:
    """第 i 行是否停车。"""
    spd = df.at[i, "GPSpdValue"]
    if pd.isna(spd):
        return True
    return float(spd) <= STOP_SPEED_KMH


def _recovers_soon(df: pd.DataFrame, i1: int, target: float) -> bool:
    """陡降后短时间内若油量回升，视为传感器尖峰而非流失。"""
    t1 = pd.Timestamp(df.at[i1, "GPSTime"])
    j = i1 + 1
    while j < len(df):
        t = pd.Timestamp(df.at[j, "GPSTime"])
        if (t - t1).total_seconds() > STEEP_DROP_RECOVER_S:
            break
        if float(df.at[j, "oilValue"]) >= target:
            return True
        j += 1
    return False


def _overlaps_any(event: RefuelEvent, others: list[RefuelEvent]) -> bool:
    """事件是否与列表中任一事件时间相交。"""
    return _overlaps_any_times(event.start_time, event.end_time, others)


def _overlaps_any_times(t0: datetime, t1: datetime, events: list[RefuelEvent]) -> bool:
    """时间窗是否与加油事件相交。"""
    a0, a1 = pd.Timestamp(t0), pd.Timestamp(t1)
    for event in events:
        b0, b1 = pd.Timestamp(event.start_time), pd.Timestamp(event.end_time)
        if a0 <= b1 and a1 >= b0:
            return True
    return False
