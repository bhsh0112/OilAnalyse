"""加油检测与油量平衡的合成数据测试。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.fuel_mgmt.load import derive_features
from src.fuel_mgmt.refuel import compute_consumption, detect_refuel_events


def _trace(rows: list[dict]) -> pd.DataFrame:
    """把简化行补全为 derive_features 所需列。"""
    t0 = datetime(2014, 11, 1, 8, 0, 0)
    records = []
    for i, r in enumerate(rows):
        rec = {
            "simNo": 1,
            "state4": 2,
            "GPSlat": r.get("lat", 43.786),
            "GPSlng": r.get("lng", 125.183),
            "BDlat": r.get("lat", 43.786),
            "BDlng": r.get("lng", 125.183),
            "GPSpdValue": r.get("spd", 0.0),
            "direValue": 0,
            "recTime": t0 + timedelta(seconds=30 * i),
            "GPSTime": t0 + timedelta(seconds=30 * i),
            "mileageValue": r.get("mile", 10000 + i),
            "oilValue": r["oil"],
            "ADValue": 500,
        }
        if "t" in r:
            rec["GPSTime"] = r["t"]
            rec["recTime"] = r["t"]
        records.append(rec)
    return derive_features(pd.DataFrame(records))


def test_parking_jitter_is_not_refuel() -> None:
    """停车 ±5 L 抖动不应识别为加油。"""
    oils = [100, 103, 98, 102, 97, 101, 96, 100, 104, 99, 100]
    df = _trace([{"oil": o, "spd": 0.0} for o in oils])
    events = detect_refuel_events(df, sigma=5.0)
    assert events == []


def test_continuous_rise_merges_to_one_refuel() -> None:
    """停车后约 10 分钟内从 80 L 升到 320 L，应合并为 1 次、约 240 L。"""
    oils = [80.0]
    # 20 个点 × 30 s ≈ 10 min，均匀升到 320
    n = 20
    for i in range(1, n + 1):
        oils.append(80.0 + 240.0 * i / n)
    df = _trace([{"oil": o, "spd": 0.0} for o in oils])
    events = detect_refuel_events(df, sigma=5.0)
    assert len(events) == 1
    ev = events[0]
    assert abs(ev.volume_l - 240.0) < 1.0
    assert ev.oil_start == 80.0
    assert ev.oil_end == 320.0
    assert ev.is_full is True


def test_balanced_consumption_formula() -> None:
    """期初 100 + 加油 200 − 期末 150 = 耗油 150。"""
    # 构造：100 停留 → 加油到 300 → 再降到 150
    oils = [100.0, 100.0]
    for x in [140, 180, 220, 260, 300]:
        oils.append(float(x))
    oils.extend([280.0, 220.0, 180.0, 150.0])
    miles = list(range(10000, 10000 + len(oils)))
    df = _trace([{"oil": o, "spd": 0.0, "mile": m} for o, m in zip(oils, miles)])
    events = detect_refuel_events(df, sigma=5.0)
    cons = compute_consumption(df, events)
    assert cons["oil_first"] == 100.0
    assert cons["oil_last"] == 150.0
    assert abs(cons["total_refuel_l"] - 200.0) < 1.0
    assert abs(cons["consumption_balanced_l"] - 150.0) < 1.0
