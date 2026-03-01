"""Sonuç gönderme sonrası KPI güncelleme ve pending_result temizleme. main.py _submit_competition_step kullanır."""
from typing import Dict, Optional, Tuple


def apply_send_result_status(
    send_status,
    pending_result: Optional[Dict],
    kpi_counters: Dict[str, int],
) -> Tuple[Optional[Dict], bool, bool]:
    status_value = str(getattr(send_status, "value", send_status))
    if status_value == "acked":
        kpi_counters["send_ok"] += 1
        return None, False, True
    if status_value == "fallback_acked":
        kpi_counters["send_ok"] += 1
        kpi_counters["send_fallback_ok"] += 1
        return None, False, True
    if status_value == "permanent_rejected":
        kpi_counters["send_fail"] += 1
        kpi_counters["send_permanent_reject"] += 1
        return None, False, False

    kpi_counters["send_fail"] += 1
    return pending_result, False, False
