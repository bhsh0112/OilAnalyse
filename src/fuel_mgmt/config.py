"""分析流水线中的阈值与常量。

第一版把可调参数集中于此，后续完善时只改这一处。
"""

from __future__ import annotations

# 停车判定：速度不超过该值视为停车（km/h）
STOP_SPEED_KMH = 1.0

# 加油候选：速度不超过该值（km/h）
REFUEL_MAX_SPEED_KMH = 5.0

# 连续上升合并的最大时间间隔（秒）
REFUEL_MERGE_GAP_S = 180.0

# 单次加油量下限（升）；最终阈值取 max(该值, 4 * 停车噪声 sigma)
REFUEL_MIN_VOLUME_L = 25.0
REFUEL_SIGMA_MULT = 4.0

# 视为有效上升的最小单步油量变化（升），用于忽略微抖动
OIL_RISE_EPS_L = 1.0

# 同一加油事件内允许的最大位移（米）
REFUEL_MAX_DISPLACEMENT_M = 300.0

# 油箱饱和读数与“疑似加满”判定
FULL_TANK_L = 320.0
FULL_MARK_L = 315.0

# 长停车最短时长（秒）
LONG_STOP_MIN_S = 30 * 60

# 驻车油量异常：净下降超过 max(该值, 3 * sigma) 列入核查
PARKING_DROP_L = 15.0
PARKING_SIGMA_MULT = 3.0

# 短时陡降（疑似偷油/漏油）：窗口内净下降超过该值且车辆停车
STEEP_DROP_WINDOW_S = 15 * 60
STEEP_DROP_L = 20.0
STEEP_DROP_SIGMA_MULT = 5.0
STEEP_DROP_RECOVER_S = 5 * 60

# AD 交叉验证：尖峰判定
AD_SPIKE_ABS = 1500.0
AD_SPIKE_ABOVE_MEDIAN = 800.0
AD_DECREASE_MIN = 20.0

# 加油间隔段油耗：里程必须大于该值才计算百公里油耗
MIN_SEGMENT_KM = 50.0

# 里程字段单位：mileageValue 的 1 个计数 = 0.1 km
MILEAGE_UNIT_KM = 0.1

# DeepSeek（OpenAI 兼容）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
API_KEY_FILENAME = "deepseek-api.txt"

# 高德 / 周边加油站（空间旁证，不作否决）
GAODE_KEY_FILENAME = "gaode-api.txt"
GAODE_AROUND_URL = "https://restapi.amap.com/v3/place/around"
GAODE_KEYWORDS = "加油站"
GAODE_TYPES = "010100|010101|010102|010103"
STATION_SEARCH_RADIUS_M = 5000
STATION_NEAR_M = 800
STATION_MID_M = 2000
OSM_OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
OSM_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSM_USER_AGENT = "fuel-mgmt-homework/1.0 (course assignment)"
