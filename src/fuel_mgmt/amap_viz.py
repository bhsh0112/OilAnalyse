"""用高德 JS API 生成带底图的轨迹 HTML。

当前 Key 是 JS 端，REST 周边检索不可用，但浏览器里铺底图可以。
产物写在 outputs/，不要把带 Key 的 HTML 提交到 git。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .refuel import RefuelEvent, haversine_m
from .stations import load_gaode_key


def _load_gaode_security() -> str:
    """读取高德 JS 安全密钥（可选，写在 gaode-api.txt 第二行或环境变量）。"""
    import os

    code = os.getenv("GAODE_SECURITY_JS_CODE", "").strip()
    if code:
        return code
    root = Path(__file__).resolve().parents[2]
    path = root / "api" / "gaode-api.txt"
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return lines[1].strip() if len(lines) > 1 else ""

CONF_COLOR = {"高可信": "#1B7F4E", "较可信": "#D97706", "存疑": "#C0392B"}


def write_trajectory_map(
    df: pd.DataFrame,
    events: Iterable[RefuelEvent],
    out_path: Path,
    api_key: str | None = None,
) -> Path:
    """写出可在浏览器打开的高德轨迹页。

    Args:
        df: 轨迹表。
        events: 加油事件。
        out_path: HTML 输出路径。
        api_key: 高德 JS Key；缺省时自动读取。

    Returns:
        写出的 HTML 路径。
    """
    key = (api_key if api_key is not None else load_gaode_key()).strip()
    payload = {
        "key": key,
        "security": _load_gaode_security(),
        "line": _downsample_line(df),
        "oilDots": _oil_dots(df),
        "events": [_event_js(e, i) for i, e in enumerate(events, 1)],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = _TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _downsample_line(df: pd.DataFrame, min_dist_m: float = 80.0, max_n: int = 2800) -> list[list[float]]:
    """按位移抽稀折线，高德坐标为 [经度, 纬度]。"""
    lats = df["GPSlat"].to_numpy(dtype=float)
    lngs = df["GPSlng"].to_numpy(dtype=float)
    line: list[list[float]] = []
    last_lat = last_lng = None
    for lat, lng in zip(lats, lngs):
        if not np.isfinite(lat) or not np.isfinite(lng):
            continue
        if last_lat is None or haversine_m(last_lat, last_lng, lat, lng) >= min_dist_m:
            line.append([round(float(lng), 6), round(float(lat), 6)])
            last_lat, last_lng = float(lat), float(lng)
    if len(line) > max_n:
        idx = np.linspace(0, len(line) - 1, max_n).astype(int)
        line = [line[i] for i in idx]
    return line


def _oil_dots(df: pd.DataFrame, max_n: int = 700) -> list[dict[str, Any]]:
    """油量着色点，比折线更稀。"""
    n = len(df)
    if n == 0:
        return []
    step = max(1, n // max_n)
    dots = []
    for i in range(0, n, step):
        lat = float(df["GPSlat"].iloc[i])
        lng = float(df["GPSlng"].iloc[i])
        if not np.isfinite(lat) or not np.isfinite(lng):
            continue
        dots.append(
            {
                "lng": round(lng, 6),
                "lat": round(lat, 6),
                "oil": round(float(df["oilValue"].iloc[i]), 1),
            }
        )
    return dots


def _event_js(event: RefuelEvent, idx: int) -> dict[str, Any]:
    """单个加油点的前端数据。"""
    t0 = event.start_time
    if not isinstance(t0, datetime):
        t0 = pd.Timestamp(t0).to_pydatetime()
    return {
        "i": idx,
        "lng": round(float(event.lng), 6),
        "lat": round(float(event.lat), 6),
        "volume": round(float(event.volume_l), 1),
        "confidence": event.confidence or "存疑",
        "color": CONF_COLOR.get(event.confidence or "存疑", "#C0392B"),
        "time": t0.strftime("%Y-%m-%d %H:%M"),
        "oilStart": round(float(event.oil_start), 1),
        "oilEnd": round(float(event.oil_end), 1),
        "station": event.station_name or "",
        "stationM": None if event.station_distance_m is None else round(float(event.station_distance_m), 0),
    }


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>营运车辆轨迹 · 高德底图</title>
  <style>
    html, body { margin: 0; height: 100%; font-family: "Microsoft YaHei", sans-serif; }
    #map { position: absolute; inset: 0; }
    .panel {
      position: absolute; top: 12px; left: 12px; z-index: 2;
      background: rgba(255,255,255,.94); border-radius: 10px;
      padding: 12px 14px; width: 280px; box-shadow: 0 6px 24px rgba(0,0,0,.12);
    }
    .panel h1 { margin: 0 0 8px; font-size: 16px; color: #1B3A4B; }
    .panel p, .legend { margin: 0; font-size: 12px; color: #4B5563; line-height: 1.55; }
    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
    .ev {
      margin-top: 8px; max-height: 46vh; overflow: auto; font-size: 12px;
    }
    .ev button {
      display: block; width: 100%; text-align: left; border: 0; background: #F3F4F6;
      margin: 0 0 6px; padding: 7px 8px; border-radius: 6px; cursor: pointer;
    }
    .warn { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
      background: #F7F5F0; color: #1B3A4B; font-size: 18px; z-index: 3; padding: 24px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel" id="panel">
    <h1>吉 A8K650 轨迹</h1>
    <p>高德底图 · GPS 轨迹抽稀后叠加。圈点颜色表示当时油量，星标为加油候选。</p>
    <p class="legend" style="margin-top:8px">
      <span class="dot" style="background:#1B7F4E"></span>高可信
      <span class="dot" style="background:#D97706"></span>较可信
      <span class="dot" style="background:#C0392B"></span>存疑
    </p>
    <div class="ev" id="ev"></div>
  </div>
  <script>
    const DATA = __DATA__;
    function oilColor(v) {
      const t = Math.max(0, Math.min(1, (v - 80) / 240));
      const r = Math.round(68 + t * 180);
      const g = Math.round(1 + t * 140);
      const b = Math.round(84 + (1 - t) * 90);
      return "rgb(" + r + "," + g + "," + b + ")";
    }
    function boot() {
      if (!DATA.key) {
        document.body.insertAdjacentHTML("beforeend",
          '<div class="warn">未找到高德 JS Key（api/gaode-api.txt）。无法铺底图。</div>');
        return;
      }
      if (DATA.security) {
        window._AMapSecurityConfig = { securityJsCode: DATA.security };
      }
      const s = document.createElement("script");
      s.src = "https://webapi.amap.com/maps?v=2.0&key=" + encodeURIComponent(DATA.key);
      s.onload = draw;
      s.onerror = function () {
        document.body.insertAdjacentHTML("beforeend",
          '<div class="warn">高德 JS API 加载失败，请检查网络与 Key。</div>');
      };
      document.head.appendChild(s);
    }
    function draw() {
      const map = new AMap.Map("map", { viewMode: "2D", zoom: 6 });
      addLayers(map);
      if (typeof AMap.plugin === "function") {
        AMap.plugin(["AMap.Scale", "AMap.ToolBar", "AMap.MapType"], function () {
          try { map.addControl(new AMap.Scale()); } catch (e) {}
          try { map.addControl(new AMap.ToolBar({ position: "RB" })); } catch (e) {}
          try { map.addControl(new AMap.MapType({ defaultType: 0 })); } catch (e) {}
        });
      }
    }
    function addLayers(map) {
      if (DATA.line && DATA.line.length) {
        const pl = new AMap.Polyline({
          path: DATA.line,
          strokeColor: "#1B3A4B",
          strokeWeight: 3,
          strokeOpacity: 0.75,
          lineJoin: "round",
          lineCap: "round"
        });
        map.add(pl);
      }
      DATA.oilDots.forEach(function (d) {
        map.add(new AMap.CircleMarker({
          center: [d.lng, d.lat],
          radius: 4,
          fillColor: oilColor(d.oil),
          fillOpacity: 0.7,
          strokeWeight: 0
        }));
      });
      const markers = [];
      DATA.events.forEach(function (e) {
        const marker = new AMap.Marker({
          position: [e.lng, e.lat],
          offset: new AMap.Pixel(-9, -9),
          content: '<div style="width:18px;height:18px;border-radius:50%;border:2px solid #fff;background:'
            + e.color + ';box-shadow:0 1px 4px rgba(0,0,0,.35)"></div>',
          title: e.i + " " + e.confidence
        });
        const station = e.station
          ? ("<br>旁证：" + e.station + (e.stationM != null ? " " + e.stationM + " m" : ""))
          : "";
        const info = new AMap.InfoWindow({
          offset: new AMap.Pixel(0, -12),
          content: "<b>#" + e.i + " " + e.confidence + "</b><br>"
            + e.time + "<br>加油 " + e.volume + " L（" + e.oilStart + "→" + e.oilEnd + "）"
            + station
        });
        marker.on("click", function () { info.open(map, marker.getPosition()); });
        map.add(marker);
        markers.push({ marker: marker, info: info, e: e });
      });
      const box = document.getElementById("ev");
      markers.forEach(function (item) {
        const b = document.createElement("button");
        b.textContent = "#" + item.e.i + "  " + item.e.time.slice(5) + "  "
          + item.e.volume + " L  " + item.e.confidence;
        b.style.borderLeft = "4px solid " + item.e.color;
        b.onclick = function () {
          map.setZoomAndCenter(12, [item.e.lng, item.e.lat]);
          item.info.open(map, item.marker.getPosition());
        };
        box.appendChild(b);
      });
      if (DATA.line && DATA.line.length) {
        map.setFitView(null, false, [60, 60, 60, 320]);
      }
    }
    boot();
  </script>
</body>
</html>
"""
