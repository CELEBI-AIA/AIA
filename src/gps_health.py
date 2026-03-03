"""GPS health parsing helpers with tri-state semantics."""

from typing import Any, Optional, Tuple


GPS_HEALTH_UNKNOWN = "unknown"
GPS_HEALTH_UNHEALTHY = "0"
GPS_HEALTH_HEALTHY = "1"


def normalize_gps_health(
    gps_health: Any,
    gps_health_status: Any = None,
) -> Tuple[Optional[int], str]:
    """Return (value, state) where value is 0/1/None and state is '0'/'1'/'unknown'."""
    raw = gps_health if gps_health is not None else gps_health_status

    if raw is None:
        return None, GPS_HEALTH_UNKNOWN

    text = str(raw).strip().lower()
    if text in {"", "unknown", "none", "null", "nan"}:
        return None, GPS_HEALTH_UNKNOWN

    try:
        parsed = int(float(raw))
    except (TypeError, ValueError):
        return None, GPS_HEALTH_UNKNOWN

    if parsed == 1:
        return 1, GPS_HEALTH_HEALTHY
    if parsed == 0:
        return 0, GPS_HEALTH_UNHEALTHY
    return None, GPS_HEALTH_UNKNOWN

