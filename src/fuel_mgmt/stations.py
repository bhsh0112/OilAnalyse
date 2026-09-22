"""加油点周边加油站检索：高德主查，失败时回退 OSM。

空间结果只作旁证：不因近站把 AD 尖峰上调，也不因无站否决高可信加油。
2014 年轨迹对照现势 POI，只能印证、不能单独定罪。
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .config import (
    GAODE_AROUND_URL,
    GAODE_KEY_FILENAME,
    GAODE_KEYWORDS,
    GAODE_TYPES,
    OSM_NOMINATIM_URL,
    OSM_OVERPASS_URLS,
    OSM_USER_AGENT,
    STATION_MID_M,
    STATION_NEAR_M,
    STATION_SEARCH_RADIUS_M,
)
from .refuel import RefuelEvent, haversine_m


def load_gaode_key() -> str:
    """读取高德 Web 服务 Key。

    优先环境变量 `GAODE_API_KEY`，否则读取 `api/gaode-api.txt`。
    """
    key = os.getenv("GAODE_API_KEY", "").strip()
    if key:
        return key
    root = Path(__file__).resolve().parents[2]
    for path in (
        root / "api" / GAODE_KEY_FILENAME,
        root / GAODE_KEY_FILENAME,
        Path.cwd() / "api" / GAODE_KEY_FILENAME,
        Path.cwd() / GAODE_KEY_FILENAME,
    ):
        if path.exists():
            return path.read_text(encoding="utf-8-sig").strip()
    return ""


def station_band(distance_m: float | None) -> str:
    """按距离划分 near / mid / far / none。"""
    if distance_m is None:
        return "none"
    if distance_m <= STATION_NEAR_M:
        return "near"
    if distance_m <= STATION_MID_M:
        return "mid"
    if distance_m <= STATION_SEARCH_RADIUS_M:
        return "far"
    return "none"


def annotate_stations(
    df,
    events: list[RefuelEvent],
    cache_path: Path | None = None,
    client: "StationClient | None" = None,
) -> list[RefuelEvent]:
    """为每个加油事件写入最近加油站名称与距离。

    Args:
        df: 轨迹表，用于读取北斗坐标。
        events: 加油事件。
        cache_path: JSON 缓存路径，避免重复打 API。
        client: 可注入的检索客户端（测试用）。

    Returns:
        原地写入空间字段后的同一批事件。
    """
    worker = client or StationClient(cache_path=cache_path)
    if client is None:
        worker.annotate_events(df, events)
        return events
    for event in events:
        bd_lat, bd_lng = _event_bd(df, event)
        hit = worker.nearest_station(event.lat, event.lng, bd_lat, bd_lng)
        _apply_hit(event, hit, worker.last_status, worker.last_source)
    return events


def spatial_effect(events: list[RefuelEvent]) -> dict[str, Any]:
    """汇总空间旁证与其对可信度的改判次数。"""
    bands = {"near": 0, "mid": 0, "far": 0, "none": 0, "error": 0, "": 0}
    changed: list[dict[str, Any]] = []
    for event in events:
        bands[event.station_band] = bands.get(event.station_band, 0) + 1
        if event.confidence_ad and event.confidence_ad != event.confidence:
            changed.append(
                {
                    "time": event.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "from": event.confidence_ad,
                    "to": event.confidence,
                    "station": event.station_name,
                    "distance_m": event.station_distance_m,
                }
            )
    usable_near = sum(
        1
        for e in events
        if e.confidence in ("高可信", "较可信") and e.station_band == "near"
    )
    return {
        "provider": next((e.station_source for e in events if e.station_source), ""),
        "query_status": next((e.station_query_status for e in events if e.station_query_status), ""),
        "bands": {k: v for k, v in bands.items() if k and v},
        "usable_near_n": usable_near,
        "n_confidence_changed": len(changed),
        "changed": changed,
        "near_m": STATION_NEAR_M,
        "mid_m": STATION_MID_M,
        "search_radius_m": STATION_SEARCH_RADIUS_M,
    }


class StationClient:
    """高德周边检索 + OSM Overpass 回退，带磁盘缓存。"""

    def __init__(self, cache_path: Path | None = None, gaode_key: str | None = None):
        """初始化客户端。"""
        self.cache_path = cache_path
        self.cache = _load_cache(cache_path)
        self.gaode_key = gaode_key if gaode_key is not None else load_gaode_key()
        self.gaode_disabled_reason = ""
        self.last_status = ""
        self.last_source = ""
        self._osm_cooldown_until = 0.0

    def annotate_events(self, df, events: list[RefuelEvent]) -> None:
        """批量标注：先探测高德，失败则用 OSM Nominatim。"""
        if not events:
            return
        print(f"检索周边加油站：{len(events)} 个候选点…", flush=True)
        if self.gaode_key:
            _hits, note = self._search_gaode(events[0].lat, events[0].lng)
            if note.startswith("ok") or note.startswith("empty"):
                for event in events:
                    bd_lat, bd_lng = _event_bd(df, event)
                    hit = self.nearest_station(event.lat, event.lng, bd_lat, bd_lng)
                    _apply_hit(event, hit, self.last_status, self.last_source)
                return
            self.gaode_disabled_reason = note
            print(f"高德周边检索不可用（{note}），改用 OSM Nominatim。", flush=True)
        else:
            self.gaode_disabled_reason = "error:no_key"

        for i, event in enumerate(events, 1):
            refs = [(event.lat, event.lng)]
            bd_lat, bd_lng = _event_bd(df, event)
            if bd_lat is not None and bd_lng is not None:
                refs.append((bd_lat, bd_lng))

            best: dict[str, Any] | None = None
            last_note = "empty"
            for qlat, qlng in refs:
                hits, last_note = self._search_nominatim(qlat, qlng)
                cand = _nearest_of(hits, refs)
                if cand is not None and (best is None or cand["distance_m"] < best["distance_m"]):
                    best = cand
                if best is not None and best["distance_m"] <= STATION_NEAR_M:
                    break
            _apply_hit(event, best, "ok" if best is not None else last_note, "osm")
            dist_txt = "—" if best is None else "{:.0f} m".format(best["distance_m"])
            name_txt = (best or {}).get("name", "未找到")
            print(
                f"  [{i}/{len(events)}] {event.start_time:%m-%d %H:%M} {name_txt} {dist_txt}",
                flush=True,
            )
        print("OSM Nominatim 检索结束。", flush=True)

    def nearest_station(
        self,
        lat: float,
        lng: float,
        bd_lat: float | None = None,
        bd_lng: float | None = None,
    ) -> dict[str, Any] | None:
        """在 GPS / 北斗周围找最近加油站。

        距离取「到 GPS 或到北斗」的较小值，以吸收约 1.3 km 系统偏差。
        """
        refs = [(lat, lng)]
        if bd_lat is not None and bd_lng is not None:
            refs.append((bd_lat, bd_lng))

        query_pts = _query_points(lat, lng, bd_lat, bd_lng)
        best: dict[str, Any] | None = None
        status_notes: list[str] = []
        source_used = ""

        for _label, qlat, qlng in query_pts:
            hits, src, note = self._search_one(qlat, qlng)
            status_notes.append(note)
            if src:
                source_used = src
            cand = _nearest_of(hits, refs)
            if cand is not None and (best is None or cand["distance_m"] < best["distance_m"]):
                best = cand
            if best is not None and best["distance_m"] <= STATION_NEAR_M:
                break

        self.last_source = (best or {}).get("source") or source_used
        if best is None:
            if any(n.startswith("error") for n in status_notes) and not any(
                n.startswith("ok") or n.startswith("empty") for n in status_notes
            ):
                self.last_status = status_notes[-1] if status_notes else "error"
            else:
                self.last_status = "empty"
            return None
        self.last_status = "ok"
        return best

    def _search_one(self, lat: float, lng: float) -> tuple[list[dict[str, Any]], str, str]:
        """单点检索：先高德，失败则 OSM。"""
        if not self.gaode_disabled_reason:
            hits, note = self._search_gaode(lat, lng)
            if note.startswith("ok") or note.startswith("empty"):
                return hits, "gaode", note
            if "PLAT_NOMATCH" in note or "USERKEY" in note or "INVALID_USER_KEY" in note:
                self.gaode_disabled_reason = note
        hits, note = self._search_osm(lat, lng)
        return hits, "osm", note

    def _search_gaode(self, lat: float, lng: float) -> tuple[list[dict[str, Any]], str]:
        """调用高德地点周边搜索。location 为 经度,纬度。"""
        if not self.gaode_key:
            return [], "error:no_key"
        cache_key = f"gaode:{lat:.5f},{lng:.5f}:{STATION_SEARCH_RADIUS_M}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached.get("hits") or [], cached.get("note") or "ok:cache"

        params = {
            "key": self.gaode_key,
            "location": f"{lng:.6f},{lat:.6f}",
            "keywords": GAODE_KEYWORDS,
            "types": GAODE_TYPES,
            "radius": str(STATION_SEARCH_RADIUS_M),
            "offset": "10",
            "page": "1",
            "extensions": "base",
            "output": "json",
        }
        url = GAODE_AROUND_URL + "?" + urllib.parse.urlencode(params)
        try:
            raw = _http_get_json(url, timeout=12)
        except Exception as exc:
            note = f"error:{type(exc).__name__}"
            self._store(cache_key, {"hits": [], "note": note})
            return [], note

        if str(raw.get("status")) != "1":
            info = str(raw.get("info") or raw.get("infocode") or "fail")
            note = f"error:{info}"
            self._store(cache_key, {"hits": [], "note": note})
            return [], note

        pois = raw.get("pois") or []
        if isinstance(pois, str):
            pois = []
        hits = []
        for poi in pois:
            loc = str(poi.get("location") or "")
            parts = loc.split(",")
            if len(parts) != 2:
                continue
            try:
                plng, plat = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            name = str(poi.get("name") or "加油站")
            hits.append({"name": name, "lat": plat, "lng": plng, "source": "gaode"})
        note = "ok" if hits else "empty"
        self._store(cache_key, {"hits": hits, "note": note})
        return hits, note

    def _search_nominatim(self, lat: float, lng: float) -> tuple[list[dict[str, Any]], str]:
        """用 Nominatim 在视窗内搜 amenity=fuel。"""
        cache_key = f"nominatim:{lat:.5f},{lng:.5f}:{STATION_SEARCH_RADIUS_M}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached.get("hits") or [], cached.get("note") or "ok:cache"

        params = {
            "format": "json",
            "limit": "8",
            "bounded": "1",
            "viewbox": _nominatim_viewbox(lat, lng),
            "amenity": "fuel",
        }
        url = OSM_NOMINATIM_URL + "?" + urllib.parse.urlencode(params)
        self._wait_osm()
        try:
            raw = _http_get_json(url, timeout=15)
        except Exception as exc:
            note = f"error:{type(exc).__name__}"
            self._store(cache_key, {"hits": [], "note": note})
            return [], note

        if not isinstance(raw, list):
            note = "error:bad_payload"
            self._store(cache_key, {"hits": [], "note": note})
            return [], note

        hits: list[dict[str, Any]] = []
        for item in raw:
            try:
                plat = float(item.get("lat"))
                plng = float(item.get("lon"))
            except (TypeError, ValueError):
                continue
            display = str(item.get("display_name") or item.get("name") or "OSM加油站")
            name = display.split(",")[0].strip() or "OSM加油站"
            hits.append({"name": name, "lat": plat, "lng": plng, "source": "osm"})
        note = "ok" if hits else "empty"
        self._store(cache_key, {"hits": hits, "note": note})
        return hits, note

    def _search_osm(self, lat: float, lng: float) -> tuple[list[dict[str, Any]], str]:
        """Overpass 单点检索 amenity=fuel。"""
        cache_key = f"osm:{lat:.5f},{lng:.5f}:{STATION_SEARCH_RADIUS_M}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached.get("hits") or [], cached.get("note") or "ok:cache"
        query = _overpass_around_query([(lat, lng)])
        return self._overpass(query, cache_key, timeout=20)

    def _search_osm_batch(
        self, points: list[tuple[float, float]]
    ) -> tuple[list[dict[str, Any]], str]:
        """一次 Overpass 查询所有候选点周围的加油站。"""
        uniq = _unique_points(points)
        cache_key = "osm_batch:" + ";".join(f"{lat:.5f},{lng:.5f}" for lat, lng in uniq)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached.get("hits") or [], cached.get("note") or "ok:cache"
        query = _overpass_around_query(uniq)
        return self._overpass(query, cache_key, timeout=70)

    def _overpass(
        self, query: str, cache_key: str, timeout: int
    ) -> tuple[list[dict[str, Any]], str]:
        """调用 Overpass 镜像并写入缓存。"""
        last_err = "error:osm"
        for url in OSM_OVERPASS_URLS:
            self._wait_osm()
            try:
                raw = _http_post_form(
                    url,
                    {"data": query},
                    timeout=timeout,
                    headers={"User-Agent": OSM_USER_AGENT},
                )
            except Exception as exc:
                last_err = f"error:{type(exc).__name__}"
                continue
            hits = _parse_overpass_elements(raw.get("elements") or [])
            note = "ok" if hits else "empty"
            self._store(cache_key, {"hits": hits, "note": note})
            return hits, note
        self._store(cache_key, {"hits": [], "note": last_err})
        return [], last_err

    def _wait_osm(self) -> None:
        """Overpass 限速：两次请求至少间隔 1 秒。"""
        now = time.time()
        wait = self._osm_cooldown_until - now
        if wait > 0:
            time.sleep(wait)
        self._osm_cooldown_until = time.time() + 1.0

    def _store(self, key: str, value: dict[str, Any]) -> None:
        """写入内存与磁盘缓存。"""
        self.cache[key] = value
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _apply_hit(
    event: RefuelEvent,
    hit: dict[str, Any] | None,
    status: str,
    source: str,
) -> None:
    """把检索结果写进事件。"""
    event.station_query_status = status
    event.station_source = source if hit or source else event.station_source
    if hit is None:
        event.station_name = None
        event.station_distance_m = None
        event.station_band = "error" if str(status).startswith("error") else "none"
        return
    event.station_name = str(hit["name"])
    event.station_distance_m = float(hit["distance_m"])
    event.station_source = str(hit.get("source") or source)
    event.station_band = station_band(event.station_distance_m)


def _event_bd(df, event: RefuelEvent) -> tuple[float | None, float | None]:
    """读取事件起点的北斗坐标。"""
    if df is None or "BDlat" not in df.columns:
        return None, None
    if not (0 <= event.start_idx < len(df)):
        return None, None
    return _as_float(df.at[event.start_idx, "BDlat"]), _as_float(df.at[event.start_idx, "BDlng"])


def _nearest_of(
    hits: list[dict[str, Any]], refs: list[tuple[float, float]]
) -> dict[str, Any] | None:
    """在 hits 中找离任一参考点最近的加油站。"""
    best: dict[str, Any] | None = None
    for hit in hits:
        dist = min(haversine_m(r[0], r[1], hit["lat"], hit["lng"]) for r in refs)
        cand = {
            "name": hit["name"],
            "lat": hit["lat"],
            "lng": hit["lng"],
            "distance_m": dist,
            "source": hit.get("source", ""),
        }
        if best is None or cand["distance_m"] < best["distance_m"]:
            best = cand
    return best


def _query_points(
    lat: float,
    lng: float,
    bd_lat: float | None,
    bd_lng: float | None,
) -> list[tuple[str, float, float]]:
    """生成检索点：原始 GPS、GCJ→WGS、北斗。"""
    pts: list[tuple[str, float, float]] = [("gps", lat, lng)]
    wgs_lat, wgs_lng = gcj02_to_wgs84(lat, lng)
    if haversine_m(lat, lng, wgs_lat, wgs_lng) > 80:
        pts.append(("gps_wgs", wgs_lat, wgs_lng))
    if bd_lat is not None and bd_lng is not None:
        pts.append(("bd", bd_lat, bd_lng))
    uniq: list[tuple[str, float, float]] = []
    for item in pts:
        if any(haversine_m(item[1], item[2], u[1], u[2]) < 40 for u in uniq):
            continue
        uniq.append(item)
    return uniq


def _nominatim_viewbox(lat: float, lng: float, radius_m: float = STATION_SEARCH_RADIUS_M) -> str:
    """Nominatim viewbox：left,top,right,bottom。"""
    dlat = radius_m / 111000.0
    clat = max(0.2, abs(math.cos(math.radians(lat))))
    dlng = radius_m / (111000.0 * clat)
    return "{:.6f},{:.6f},{:.6f},{:.6f}".format(lng - dlng, lat + dlat, lng + dlng, lat - dlat)


def _unique_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """按约 1 m 精度去重检索点。"""
    uniq: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for lat, lng in points:
        key = (round(lat, 5), round(lng, 5))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((lat, lng))
    return uniq


def _overpass_around_query(points: list[tuple[float, float]]) -> str:
    """构造 around 并集查询。"""
    clauses: list[str] = []
    for lat, lng in points:
        clauses.append(
            f'node["amenity"="fuel"](around:{STATION_SEARCH_RADIUS_M},{lat:.6f},{lng:.6f});'
        )
        clauses.append(
            f'way["amenity"="fuel"](around:{STATION_SEARCH_RADIUS_M},{lat:.6f},{lng:.6f});'
        )
    return f"[out:json][timeout:60];({''.join(clauses)});out center;"


def _parse_overpass_elements(elements: list[Any]) -> list[dict[str, Any]]:
    """把 Overpass elements 转成加油站列表。"""
    hits: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for el in elements:
        tags = el.get("tags") or {}
        elat = el.get("lat")
        elng = el.get("lon")
        if elat is None or elng is None:
            center = el.get("center") or {}
            elat, elng = center.get("lat"), center.get("lon")
        if elat is None or elng is None:
            continue
        key = (round(float(elat), 5), round(float(elng), 5))
        if key in seen:
            continue
        seen.add(key)
        name = (
            tags.get("name")
            or tags.get("name:zh")
            or tags.get("brand")
            or tags.get("operator")
            or "OSM加油站"
        )
        hits.append(
            {
                "name": str(name),
                "lat": float(elat),
                "lng": float(elng),
                "source": "osm",
            }
        )
    return hits


def wgs84_to_gcj02(lat: float, lng: float) -> tuple[float, float]:
    """WGS-84 转 GCJ-02（高德）。境外原样返回。"""
    if _out_of_china(lat, lng):
        return lat, lng
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    rad_lat = lat / 180.0 * math.pi
    magic = math.sin(rad_lat)
    magic = 1 - _EE * magic * magic
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrt_magic) * math.pi)
    dlng = (dlng * 180.0) / (_A / sqrt_magic * math.cos(rad_lat) * math.pi)
    return lat + dlat, lng + dlng


def gcj02_to_wgs84(lat: float, lng: float) -> tuple[float, float]:
    """GCJ-02 近似反算 WGS-84。"""
    glat, glng = wgs84_to_gcj02(lat, lng)
    return lat * 2 - glat, lng * 2 - glng


_A = 6378245.0
_EE = 0.00669342162296594323


def _out_of_china(lat: float, lng: float) -> bool:
    """是否明显不在国内加密范围。"""
    return not (73.66 < lng < 135.05 and 3.86 < lat < 53.55)


def _transform_lat(x: float, y: float) -> float:
    """GCJ 偏移公式：纬度分量。"""
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(x: float, y: float) -> float:
    """GCJ 偏移公式：经度分量。"""
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def _load_cache(path: Path | None) -> dict[str, Any]:
    """读取磁盘缓存。"""
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _as_float(value: Any) -> float | None:
    """把单元格转成 float；缺失返回 None。"""
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _http_get_json(url: str, timeout: int = 20) -> dict[str, Any]:
    """GET JSON。"""
    req = urllib.request.Request(url, headers={"User-Agent": OSM_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_form(
    url: str,
    fields: dict[str, str],
    timeout: int = 40,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST application/x-www-form-urlencoded 并解析 JSON。"""
    body = urllib.parse.urlencode(fields).encode("utf-8")
    hdrs = {"User-Agent": OSM_USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
