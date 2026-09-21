"""读取油耗轨迹并派生分析用字段。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import MILEAGE_UNIT_KM, STOP_SPEED_KMH


REQUIRED_COLUMNS = [
    "simNo",
    "state4",
    "GPSlat",
    "GPSlng",
    "BDlat",
    "BDlng",
    "GPSpdValue",
    "direValue",
    "recTime",
    "GPSTime",
    "mileageValue",
    "oilValue",
    "ADValue",
]


def load_trace(path: str | Path) -> pd.DataFrame:
    """读取 xls 轨迹并完成排序与特征派生。

    Args:
        path: 原始 Excel 路径（.xls）。

    Returns:
        按 GPSTime 升序、带派生列的 DataFrame。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    df = pd.read_excel(path, sheet_name=0, engine="xlrd")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少字段: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["GPSTime"] = pd.to_datetime(df["GPSTime"])
    df["recTime"] = pd.to_datetime(df["recTime"])
    return derive_features(df)


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """按 GPSTime 排序并计算间隔、油量差分、里程差分、停车标记。

    Args:
        df: 至少包含 GPSTime、oilValue、mileageValue、GPSpdValue 的表。

    Returns:
        带 dt_s / doil / dmile_km / is_stop / date 等列的新表。
    """
    out = df.copy()
    out["GPSTime"] = pd.to_datetime(out["GPSTime"])
    if "recTime" in out.columns:
        out["recTime"] = pd.to_datetime(out["recTime"])

    out = out.sort_values("GPSTime", kind="mergesort").reset_index(drop=True)
    out["dt_s"] = out["GPSTime"].diff().dt.total_seconds()
    out["doil"] = out["oilValue"].diff()
    out["dmile_km"] = out["mileageValue"].diff() * MILEAGE_UNIT_KM
    out["is_stop"] = out["GPSpdValue"].fillna(0) <= STOP_SPEED_KMH
    out["date"] = out["GPSTime"].dt.date
    out["hour"] = out["GPSTime"].dt.hour
    if "recTime" in out.columns:
        out["rec_lag_min"] = (out["recTime"] - out["GPSTime"]).dt.total_seconds() / 60.0
    else:
        out["rec_lag_min"] = pd.NA
    return out
