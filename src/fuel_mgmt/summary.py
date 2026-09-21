"""汇总给报告与大模型的结构化事实，不发送原始轨迹。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .mining import strip_stop_internal_ts
from .refuel import RefuelEvent


def build_summary(
    *,
    data_path: str,
    quality: dict[str, Any],
    events: list[RefuelEvent],
    consumption: dict[str, Any],
    daily: list[dict[str, Any]],
    stops: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    economy: list[dict[str, Any]],
    threshold_l: float,
    consumption_high: dict[str, Any] | None = None,
    consumption_usable: dict[str, Any] | None = None,
    steep_drops: list[dict[str, Any]] | None = None,
    timing: dict[str, Any] | None = None,
    confidence: dict[str, Any] | None = None,
    n_slow: int = 0,
) -> dict[str, Any]:
    """组装 analysis_summary.json 的内容。"""
    return {
        "vehicle": {
            "sim_no": quality.get("sim_no"),
            "data_file": data_path,
            "time_start": quality.get("time_start"),
            "time_end": quality.get("time_end"),
            "n_records": quality.get("n_records"),
            "mileage_km": quality.get("mileage_km"),
        },
        "sensor": {
            "oil_noise_sigma_l": round(float(quality["noise"]["sigma_l"]), 2),
            "stop_doil_p95_abs": round(float(quality["noise"]["stop"]["p95_abs"]), 2),
            "saturation": quality.get("saturation"),
            "sampling": quality.get("sampling"),
            "rec_time_lag_min": quality.get("rec_time_lag_min"),
            "naive_neg_sum_stop_l": round(float(quality["noise"]["naive_neg_sum_stop_l"]), 1),
            "naive_neg_sum_moving_l": round(float(quality["noise"]["naive_neg_sum_moving_l"]), 1),
            "ad_oil_corr": quality.get("ad_oil_corr"),
            "gnss_offset_m": quality.get("gnss_offset_m"),
        },
        "homework": {
            "refuel_threshold_l": round(float(threshold_l), 1),
            "n_refuel": consumption.get("n_refuel"),
            "total_refuel_l": consumption.get("total_refuel_l"),
            "consumption_balanced_l": consumption.get("consumption_balanced_l"),
            "consumption_naive_neg_sum_l": consumption.get("consumption_naive_neg_sum_l"),
            "oil_first": consumption.get("oil_first"),
            "oil_last": consumption.get("oil_last"),
            "n_slow_refuel": n_slow,
            "confidence": confidence or {},
            "high_confidence_consumption": consumption_high or {},
            "usable_consumption": consumption_usable or {},
            "events": [e.to_dict() for e in events],
        },
        "operations": {
            "daily": daily,
            "long_stops": strip_stop_internal_ts(stops),
            "parking_anomalies": anomalies,
            "steep_drops": steep_drops or [],
            "inter_refuel_economy": economy,
            "refuel_timing": timing or {},
        },
    }


def write_summary(summary: dict[str, Any], path: Path) -> None:
    """把摘要写成 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
