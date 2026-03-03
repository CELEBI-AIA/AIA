"""Versioned payload adapter for outbound competition JSON."""

from dataclasses import dataclass
from typing import Any, Dict, List

from config.settings import Settings
from src.competition_contract import DataContractError


@dataclass(frozen=True)
class PayloadProfile:
    version: str
    cls_as_int: bool
    status_type: str
    motion_field: str


class PayloadAdapter:
    """Single point for payload profile versioning and field casting."""

    _SUPPORTED = {"v1", "v1_legacy", "v2_int"}
    _CANONICAL_MOTION_FIELD = "motion_status"
    _LEGACY_MOTION_FIELD = "movement_status"

    @classmethod
    def self_check(cls) -> None:
        cls.resolve_profile()

    @classmethod
    def resolve_profile(cls, version: str | None = None) -> PayloadProfile:
        requested = (
            str(version or getattr(Settings, "PAYLOAD_ADAPTER_VERSION", "v1"))
            .strip()
            .lower()
        )
        if requested not in cls._SUPPORTED:
            raise DataContractError(
                f"Unsupported PAYLOAD_ADAPTER_VERSION='{requested}'. "
                f"Supported={sorted(cls._SUPPORTED)}"
            )

        if requested == "v1":
            return PayloadProfile(
                version="v1",
                cls_as_int=bool(getattr(Settings, "PAYLOAD_CLS_AS_INT", False)),
                status_type=str(
                    getattr(Settings, "PAYLOAD_STATUS_TYPE_PROFILE", "int")
                ).strip().lower(),
                motion_field=cls._CANONICAL_MOTION_FIELD,
            )

        if requested == "v1_legacy":
            return PayloadProfile(
                version="v1_legacy",
                cls_as_int=False,
                status_type="string",
                motion_field=cls._LEGACY_MOTION_FIELD,
            )

        return PayloadProfile(
            version="v2_int",
            cls_as_int=True,
            status_type="int",
            motion_field=cls._CANONICAL_MOTION_FIELD,
        )

    @classmethod
    def adapt_payload(
        cls,
        payload: Dict[str, Any],
        version: str | None = None,
    ) -> Dict[str, Any]:
        profile = cls.resolve_profile(version=version)
        objects_raw = payload.get("detected_objects", [])
        objects: List[Dict[str, Any]] = []

        if isinstance(objects_raw, list):
            for obj in objects_raw:
                if not isinstance(obj, dict):
                    continue
                objects.append(cls._adapt_object(obj, profile))

        adapted = {
            "id": payload.get("id"),
            "user": payload.get("user"),
            "frame": payload.get("frame"),
            "detected_objects": objects,
            "detected_translations": payload.get("detected_translations", []),
            "detected_undefined_objects": payload.get(
                "detected_undefined_objects", []
            ),
        }
        return adapted

    @classmethod
    def _adapt_object(cls, obj: Dict[str, Any], profile: PayloadProfile) -> Dict[str, Any]:
        class_value = cls._safe_int(obj.get("cls", -1), default=-1)
        out_class: Any = class_value if profile.cls_as_int else str(class_value)

        landing = cls._cast_status(obj.get("landing_status", -1), profile.status_type)
        motion_raw = obj.get(cls._CANONICAL_MOTION_FIELD, obj.get(cls._LEGACY_MOTION_FIELD, -1))
        motion = cls._cast_status(motion_raw, profile.status_type)

        adapted: Dict[str, Any] = {
            "cls": out_class,
            "landing_status": landing,
            "top_left_x": cls._safe_int(obj.get("top_left_x", 0), default=0),
            "top_left_y": cls._safe_int(obj.get("top_left_y", 0), default=0),
            "bottom_right_x": cls._safe_int(obj.get("bottom_right_x", 1), default=1),
            "bottom_right_y": cls._safe_int(obj.get("bottom_right_y", 1), default=1),
        }
        adapted[profile.motion_field] = motion
        return adapted

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _cast_status(value: Any, status_type: str) -> Any:
        safe = PayloadAdapter._safe_int(value, default=-1)
        if status_type in {"string", "str"}:
            return str(safe)
        return int(safe)

