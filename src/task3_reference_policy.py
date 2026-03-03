"""Shared Task3 reference validation and canonicalization policy."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import PROJECT_ROOT, Settings


def normalize_task3_object_id(raw_object_id: Any) -> Optional[int]:
    if isinstance(raw_object_id, bool) or raw_object_id is None:
        return None

    if isinstance(raw_object_id, int):
        object_id = raw_object_id
    elif isinstance(raw_object_id, float):
        if not raw_object_id.is_integer():
            return None
        object_id = int(raw_object_id)
    elif isinstance(raw_object_id, str):
        stripped = raw_object_id.strip()
        if not stripped:
            return None
        try:
            object_id = int(stripped)
        except ValueError:
            return None
    else:
        return None

    if object_id < 0:
        return None
    return object_id


def build_task3_reference_source(
    ref_data: Dict[str, Any],
    object_id: int,
    project_root: Optional[str | Path] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    label = str(ref_data.get("label", f"ref_{object_id}"))

    if ref_data.get("path"):
        path_raw = str(ref_data["path"])
        path_obj = Path(path_raw)
        if not path_obj.is_absolute():
            base = Path(project_root) if project_root is not None else PROJECT_ROOT
            path_obj = (base / path_obj).resolve()
        return {"object_id": object_id, "path": str(path_obj), "label": label}, "path"

    image_base64 = ref_data.get("image_base64")
    if image_base64:
        try:
            arr = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                return None, "base64_decode_failed"
            return {"object_id": object_id, "image": image, "label": label}, "image_base64"
        except (TypeError, ValueError):
            return None, "base64_decode_failed"

    image = ref_data.get("image")
    if image is not None:
        return {"object_id": object_id, "image": image, "label": label}, "image"

    return None, "missing_image_source"


def canonicalize_task3_references(
    log: Any,
    references: List[Any],
    project_root: Optional[str | Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], str, str, bool]:
    stats: Dict[str, int] = {
        "total": len(references),
        "valid": 0,
        "duplicate": 0,
        "quarantined": 0,
    }
    canonical_refs: List[Dict[str, Any]] = []
    seen_object_ids: Dict[int, Dict[str, Any]] = {}

    for idx, raw_ref in enumerate(references):
        if not isinstance(raw_ref, dict):
            stats["quarantined"] += 1
            log.warn(
                f"event=task3_ref_quarantined reason=invalid_record_type index={idx}"
            )
            continue

        object_id = normalize_task3_object_id(raw_ref.get("object_id"))
        if object_id is None:
            stats["quarantined"] += 1
            log.warn(
                "event=task3_ref_quarantined "
                f"reason=invalid_object_id index={idx} raw_object_id={raw_ref.get('object_id')}"
            )
            continue

        source_ref, source_kind = build_task3_reference_source(
            raw_ref,
            object_id,
            project_root=project_root,
        )
        if source_ref is None:
            stats["quarantined"] += 1
            log.warn(
                "event=task3_ref_quarantined "
                f"reason={source_kind} object_id={object_id} index={idx}"
            )
            continue

        if object_id in seen_object_ids:
            stats["duplicate"] += 1
            stats["quarantined"] += 1
            first_source = seen_object_ids[object_id].get("_source_kind", "unknown")
            log.warn(
                "event=task3_ref_duplicate_detected "
                f"object_id={object_id} first_source={first_source} "
                f"duplicate_source={source_kind} index={idx}"
            )
            log.warn(
                "event=task3_ref_quarantined "
                f"reason=duplicate_object_id object_id={object_id} index={idx}"
            )
            continue

        source_ref["_source_kind"] = source_kind
        seen_object_ids[object_id] = source_ref
        canonical_refs.append(source_ref)
        stats["valid"] += 1

    for ref in canonical_refs:
        ref.pop("_source_kind", None)

    duplicate_ratio = (
        float(stats["duplicate"]) / float(stats["total"])
        if stats["total"] > 0
        else 0.0
    )
    duplicate_critical = stats["duplicate"] >= int(
        getattr(Settings, "TASK3_DUPLICATE_DEGRADE_MIN_COUNT", 3)
    ) and duplicate_ratio >= float(getattr(Settings, "TASK3_DUPLICATE_DEGRADE_RATIO", 0.5))

    if duplicate_critical:
        id_integrity_mode = "degraded"
        id_integrity_reason_code = "duplicate_ratio_critical"
    elif stats["duplicate"] > 0:
        id_integrity_mode = "degraded"
        id_integrity_reason_code = "duplicate_detected_safe_degrade"
    elif stats["quarantined"] > 0:
        id_integrity_mode = "degraded"
        id_integrity_reason_code = "reference_quarantined_non_duplicate"
    else:
        id_integrity_mode = "normal"
        id_integrity_reason_code = "ok"

    log.info(
        "event=task3_ref_validation_summary "
        f"total={stats['total']} valid={stats['valid']} duplicate={stats['duplicate']} "
        f"quarantined={stats['quarantined']}"
    )
    log.warn(
        "event=task3_id_integrity_mode "
        f"mode={id_integrity_mode} reason_code={id_integrity_reason_code} "
        f"duplicate_ratio={duplicate_ratio:.3f}"
    )

    return (
        canonical_refs,
        stats,
        id_integrity_mode,
        id_integrity_reason_code,
        duplicate_critical,
    )

