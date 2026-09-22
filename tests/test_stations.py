"""周边加油站检索与坐标变换测试（不访问外网）。"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.fuel_mgmt.refuel import RefuelEvent, haversine_m
from src.fuel_mgmt.stations import (
    annotate_stations,
    gcj02_to_wgs84,
    spatial_effect,
    station_band,
    wgs84_to_gcj02,
)


class _FakeClient:
    """返回固定近站的桩。"""

    last_status = "ok"
    last_source = "gaode"

    def nearest_station(self, lat, lng, bd_lat=None, bd_lng=None):
        """忽略坐标，返回 120 m 处加油站。"""
        return {
            "name": "中石化演示站",
            "lat": lat,
            "lng": lng,
            "distance_m": 120.0,
            "source": "gaode",
        }


def test_station_band_thresholds() -> None:
    """距离分层边界。"""
    assert station_band(None) == "none"
    assert station_band(0) == "near"
    assert station_band(800) == "near"
    assert station_band(801) == "mid"
    assert station_band(2000) == "mid"
    assert station_band(2001) == "far"
    assert station_band(5000) == "far"
    assert station_band(5001) == "none"


def test_wgs_gcj_roundtrip_in_china() -> None:
    """国内点 WGS→GCJ 应有数百米偏移，反算误差应很小。"""
    lat, lng = 39.9042, 116.4074
    glat, glng = wgs84_to_gcj02(lat, lng)
    shift = haversine_m(lat, lng, glat, glng)
    assert 200 < shift < 800
    blat, blng = gcj02_to_wgs84(glat, glng)
    assert haversine_m(lat, lng, blat, blng) < 5


def test_annotate_writes_station_fields() -> None:
    """注入客户端后应写入名称与 near 分层。"""
    t0 = datetime(2014, 11, 1, 8, 0, 0)
    event = RefuelEvent(
        start_time=t0,
        end_time=t0,
        lat=36.384,
        lng=118.637,
        oil_start=124.0,
        oil_end=320.0,
        volume_l=196.0,
        is_full=True,
        n_samples=7,
        start_idx=0,
        end_idx=1,
    )
    df = pd.DataFrame({"BDlat": [36.39], "BDlng": [118.64]})
    out = annotate_stations(df, [event], client=_FakeClient())
    assert out[0].station_name == "中石化演示站"
    assert out[0].station_distance_m == 120.0
    assert out[0].station_band == "near"
    assert out[0].station_source == "gaode"


def test_spatial_effect_counts_near_without_rerate() -> None:
    """旁证汇总：近距印证但不改判时 changed 为空。"""
    t0 = datetime(2014, 11, 1, 8, 0, 0)
    event = RefuelEvent(
        start_time=t0,
        end_time=t0,
        lat=36.384,
        lng=118.637,
        oil_start=124.0,
        oil_end=320.0,
        volume_l=196.0,
        is_full=True,
        n_samples=7,
        start_idx=0,
        end_idx=1,
        confidence="高可信",
        confidence_ad="高可信",
        station_band="near",
        station_name="演示站",
        station_distance_m=120.0,
        station_source="osm",
        station_query_status="ok",
    )
    effect = spatial_effect([event])
    assert effect["bands"]["near"] == 1
    assert effect["usable_near_n"] == 1
    assert effect["n_confidence_changed"] == 0
