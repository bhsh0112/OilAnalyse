"""加油可信度与短时陡降测试。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.fuel_mgmt.evidence import (
    apply_spatial_assist,
    detect_steep_drops,
    score_refuel_events,
)
from src.fuel_mgmt.load import derive_features
from src.fuel_mgmt.refuel import RefuelEvent, detect_refuel_events


def _trace(rows: list[dict]) -> pd.DataFrame:
    """补全轨迹列，支持自定义 ADValue。"""
    t0 = datetime(2014, 11, 1, 8, 0, 0)
    records = []
    for i, r in enumerate(rows):
        rec = {
            "simNo": 1,
            "state4": 2,
            "GPSlat": r.get("lat", 43.786),
            "GPSlng": r.get("lng", 125.183),
            "BDlat": r.get("lat", 43.786) + 0.012,
            "BDlng": r.get("lng", 125.183) + 0.012,
            "GPSpdValue": r.get("spd", 0.0),
            "direValue": 0,
            "recTime": t0 + timedelta(seconds=30 * i),
            "GPSTime": t0 + timedelta(seconds=30 * i),
            "mileageValue": r.get("mile", 10000),
            "oilValue": r["oil"],
            "ADValue": r.get("ad", 500),
        }
        records.append(rec)
    return derive_features(pd.DataFrame(records))


def test_smooth_fill_with_ad_drop_is_high_confidence() -> None:
    """油量连续加满且 AD 同步下降，应为高可信。"""
    n = 20
    rows = [{"oil": 80.0, "ad": 500, "spd": 0.0}]
    for i in range(1, n + 1):
        rows.append({"oil": 80.0 + 240.0 * i / n, "ad": 500 - 18 * i, "spd": 0.0})
    df = _trace(rows)
    events = score_refuel_events(df, detect_refuel_events(df, sigma=5.0))
    assert len(events) == 1
    assert events[0].confidence == "高可信"
    assert events[0].ad_ok is True
    assert events[0].ad_spike is False


def test_presample_ad_spike_does_not_veto_clean_rise() -> None:
    """加油前一条 AD 尖峰、上升段内正常下降，不单独判存疑。"""
    rows = [{"oil": 100.0, "ad": 2500, "spd": 0.0}]
    for i in range(1, 8):
        rows.append({"oil": 100.0 + 30.0 * i, "ad": 480.0 - 40.0 * i, "spd": 0.0})
    df = _trace(rows)
    events = score_refuel_events(df, detect_refuel_events(df, sigma=5.0))
    assert len(events) == 1
    assert events[0].ad_spike is False
    assert events[0].confidence == "高可信"
    assert any("不单独否决" in r for r in events[0].reasons)


def test_ad_spike_inside_rise_marks_suspect() -> None:
    """加油上升段内出现 AD 尖峰，应判为存疑。"""
    rows = [
        {"oil": 100.0, "ad": 400, "spd": 0.0},
        {"oil": 150.0, "ad": 2500, "spd": 0.0},
        {"oil": 200.0, "ad": 410, "spd": 0.0},
        {"oil": 250.0, "ad": 380, "spd": 0.0},
    ]
    df = _trace(rows)
    events = score_refuel_events(df, detect_refuel_events(df, sigma=5.0))
    assert len(events) == 1
    assert events[0].ad_spike is True
    assert events[0].confidence == "存疑"


def test_sustained_high_ad_is_spike_even_if_median_is_high() -> None:
    """上升段内 AD 持续 ≥1500 时，即使中位数也被抬高，仍应记为尖峰。"""
    rows = [{"oil": 80.0, "ad": 500, "spd": 0.0}]
    for i in range(1, 8):
        rows.append({"oil": 80.0 + 30.0 * i, "ad": 2000.0 - 20.0 * i, "spd": 0.0})
    df = _trace(rows)
    events = score_refuel_events(df, detect_refuel_events(df, sigma=5.0))
    assert len(events) == 1
    assert events[0].ad_spike is True
    assert events[0].confidence == "存疑"


def test_ad_reverse_during_rise_is_suspect() -> None:
    """油量连续上升但上升段 AD 同步升高，应判为存疑。"""
    rows = [{"oil": 80.0, "ad": 500, "spd": 0.0}]
    for i in range(1, 8):
        rows.append({"oil": 80.0 + 30.0 * i, "ad": 500.0 + 40.0 * i, "spd": 0.0})
    df = _trace(rows)
    events = score_refuel_events(df, detect_refuel_events(df, sigma=5.0))
    assert len(events) == 1
    assert events[0].ad_ok is False
    assert float(events[0].ad_delta) > 0
    assert events[0].confidence == "存疑"
    assert any("反向" in r for r in events[0].reasons)


def test_parking_jitter_is_not_steep_drop() -> None:
    """停车 ±5 L 抖动不应记为短时陡降。"""
    oils = [100, 103, 98, 102, 97, 101, 96, 100, 104, 99, 100]
    df = _trace([{"oil": o, "spd": 0.0, "ad": 500} for o in oils])
    drops = detect_steep_drops(df, events=[], sigma=5.0)
    assert drops == []


def test_steep_drop_while_parked() -> None:
    """停车后数分钟油量下降超过阈值，且不回升，应记为陡降。"""
    rows = [{"oil": 200.0, "ad": 200, "spd": 0.0} for _ in range(4)]
    for oil, ad in [(180, 260), (165, 310), (150, 360), (148, 370), (147, 372), (147, 372)]:
        rows.append({"oil": float(oil), "ad": ad, "spd": 0.0})
    df = _trace(rows)
    drops = detect_steep_drops(df, events=[], sigma=5.0)
    assert len(drops) >= 1
    assert drops[0]["drop_l"] >= 20


def test_ad_spike_near_station_stays_suspect() -> None:
    """AD 尖峰即使贴着加油站也不上调。"""
    rows = [
        {"oil": 100.0, "ad": 400, "spd": 0.0},
        {"oil": 150.0, "ad": 2500, "spd": 0.0},
        {"oil": 200.0, "ad": 410, "spd": 0.0},
        {"oil": 250.0, "ad": 380, "spd": 0.0},
    ]
    df = _trace(rows)
    events = detect_refuel_events(df, sigma=5.0)
    assert len(events) == 1
    events[0].station_name = "中石化演示站"
    events[0].station_distance_m = 80.0
    events[0].station_band = "near"
    events[0].station_source = "gaode"
    events[0].station_query_status = "ok"
    scored = score_refuel_events(df, events)
    assert scored[0].confidence_ad == "存疑"
    assert scored[0].confidence == "存疑"
    assert any("不上调" in r for r in scored[0].reasons)
    assert any("中石化" in r for r in scored[0].reasons)


def test_high_confidence_not_vetoed_without_station() -> None:
    """高可信在半径内无加油站时不被否决。"""
    n = 20
    rows = [{"oil": 80.0, "ad": 500, "spd": 0.0}]
    for i in range(1, n + 1):
        rows.append({"oil": 80.0 + 240.0 * i / n, "ad": 500 - 18 * i, "spd": 0.0})
    df = _trace(rows)
    events = detect_refuel_events(df, sigma=5.0)
    events[0].station_band = "none"
    events[0].station_query_status = "empty"
    scored = score_refuel_events(df, events)
    assert scored[0].confidence_ad == "高可信"
    assert scored[0].confidence == "高可信"


def test_weak_ad_probable_demoted_without_station() -> None:
    """较可信但 AD 不支持、且附近无站时，下调为存疑。"""
    t0 = datetime(2014, 11, 1, 8, 0, 0)
    event = RefuelEvent(
        start_time=t0,
        end_time=t0,
        lat=36.38,
        lng=118.63,
        oil_start=100.0,
        oil_end=200.0,
        volume_l=100.0,
        is_full=False,
        n_samples=4,
        start_idx=0,
        end_idx=3,
        confidence="较可信",
        confidence_ad="较可信",
        ad_ok=False,
        station_band="none",
        station_query_status="empty",
        reasons=["形态像加油但 AD 证据不足"],
    )
    apply_spatial_assist(event)
    assert event.confidence == "存疑"
    assert any("下调为存疑" in r for r in event.reasons)

