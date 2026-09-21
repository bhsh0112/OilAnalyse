"""数据质量与油量传感器噪声分析。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import FULL_TANK_L


def estimate_oil_noise(df: pd.DataFrame) -> dict[str, Any]:
    """用停车样本估计油量短时噪声。

    停车时油量不应系统性消耗到与行驶段同量级；该处的 doil 分布
    即传感器抖动，后续加油阈值必须明显高于它。

    Args:
        df: 已派生 is_stop、doil 的轨迹表。

    Returns:
        含 sigma、分位数及停车/行驶对照的字典。
    """
    doil = df["doil"]
    stop_mask = df["is_stop"] & doil.notna()
    move_mask = (~df["is_stop"]) & doil.notna()
    stop_doil = doil.loc[stop_mask]
    move_doil = doil.loc[move_mask]

    def _stats(s: pd.Series) -> dict[str, float]:
        if s.empty:
            return {"n": 0, "mean": 0.0, "std": 0.0, "p95_abs": 0.0}
        return {
            "n": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "p95_abs": float(s.abs().quantile(0.95)),
        }

    stop_stats = _stats(stop_doil)
    move_stats = _stats(move_doil)
    sigma = stop_stats["std"] if stop_stats["n"] > 2 else 5.0
    return {
        "sigma_l": sigma,
        "stop": stop_stats,
        "moving": move_stats,
        "naive_neg_sum_stop_l": float(-stop_doil.loc[stop_doil < 0].sum()) if not stop_doil.empty else 0.0,
        "naive_neg_sum_moving_l": float(-move_doil.loc[move_doil < 0].sum()) if not move_doil.empty else 0.0,
    }


def analyze_quality(df: pd.DataFrame) -> dict[str, Any]:
    """汇总采样、时间轴、油量饱和与噪声等质量指标。

    Args:
        df: `load_trace` / `derive_features` 的输出。

    Returns:
        可写入 JSON 的质量摘要。
    """
    dt = df["dt_s"].dropna()
    lag = df["rec_lag_min"].dropna() if "rec_lag_min" in df.columns else pd.Series(dtype=float)
    oil = df["oilValue"]
    noise = estimate_oil_noise(df)

    sat_n = int((oil >= FULL_TANK_L - 1e-6).sum())
    mileage_km = float(
        (df["mileageValue"].iloc[-1] - df["mileageValue"].iloc[0]) * 0.1
    ) if len(df) else 0.0

    ad_corr = None
    if "ADValue" in df.columns and len(df) > 2:
        ad_corr = float(df[["oilValue", "ADValue"]].corr().iloc[0, 1])

    gnss = _gnss_offset(df)

    return {
        "n_records": int(len(df)),
        "sim_no": _unique_or_first(df["simNo"]) if "simNo" in df.columns else None,
        "time_start": df["GPSTime"].min().isoformat(sep=" ") if len(df) else None,
        "time_end": df["GPSTime"].max().isoformat(sep=" ") if len(df) else None,
        "mileage_km": round(mileage_km, 1),
        "oil_min": float(oil.min()) if len(df) else None,
        "oil_max": float(oil.max()) if len(df) else None,
        "sampling": {
            "median_s": float(dt.median()) if len(dt) else None,
            "mean_s": float(dt.mean()) if len(dt) else None,
            "p99_s": float(dt.quantile(0.99)) if len(dt) else None,
            "max_s": float(dt.max()) if len(dt) else None,
            "gap_gt_120s": int((dt > 120).sum()) if len(dt) else 0,
            "gap_gt_3600s": int((dt > 3600).sum()) if len(dt) else 0,
        },
        "rec_time_lag_min": {
            "median": float(lag.median()) if len(lag) else None,
            "mean": float(lag.mean()) if len(lag) else None,
            "max": float(lag.max()) if len(lag) else None,
        },
        "saturation": {
            "full_tank_l": FULL_TANK_L,
            "n_at_full": sat_n,
            "ratio": float(sat_n / len(df)) if len(df) else 0.0,
        },
        "ad_oil_corr": ad_corr,
        "gnss_offset_m": gnss,
        "noise": noise,
    }


def _gnss_offset(df: pd.DataFrame) -> dict[str, float] | None:
    """GPS 与北斗的平面距离统计（粗算，单位米）。"""
    if not {"GPSlat", "GPSlng", "BDlat", "BDlng"} <= set(df.columns) or df.empty:
        return None
    dlat = (df["BDlat"] - df["GPSlat"]) * 111000.0
    dlng = (df["BDlng"] - df["GPSlng"]) * 111000.0 * np.cos(np.radians(df["GPSlat"]))
    dist = np.sqrt(dlat ** 2 + dlng ** 2).dropna()
    if dist.empty:
        return None
    return {
        "median_m": round(float(dist.median()), 1),
        "std_m": round(float(dist.std(ddof=1)), 1) if len(dist) > 1 else 0.0,
        "min_m": round(float(dist.min()), 1),
        "max_m": round(float(dist.max()), 1),
    }


def _unique_or_first(s: pd.Series) -> Any:
    """取唯一值；若不唯一则返回首个值。"""
    vals = s.dropna().unique()
    if len(vals) == 1:
        v = vals[0]
        return int(v) if isinstance(v, (np.integer, int)) else v
    if len(vals) == 0:
        return None
    v = vals[0]
    return int(v) if isinstance(v, (np.integer, int)) else v
