"""Canonical class contract for TEKNOFEST object classes.

Single source of truth:
- class id ordering (dataset / model / inference / payload)
- display labels
- alias resolution for model label -> class id mapping
"""

from typing import Dict, List, Tuple

from config.settings import Settings
from src.competition_contract import DataContractError


class CompetitionClassContract:
    """Global class-id contract shared across modules."""

    ORDERED_IDS: Tuple[int, ...] = (0, 1, 2, 3)
    ORDERED_KEYS: Tuple[str, ...] = ("tasit", "insan", "uap", "uai")
    DISPLAY_NAMES: Dict[int, str] = {
        0: "Tasit",
        1: "Insan",
        2: "UAP",
        3: "UAI",
    }
    _ALIASES: Dict[str, int] = {
        # Tasit
        "tasit": 0,
        "vehicle": 0,
        "car": 0,
        "van": 0,
        "truck": 0,
        "bus": 0,
        "train": 0,
        "boat": 0,
        "bicycle": 0,
        "motorcycle": 0,
        "motor": 0,
        "tricycle": 0,
        "awning tricycle": 0,
        # Insan
        "insan": 1,
        "human": 1,
        "person": 1,
        "pedestrian": 1,
        "people": 1,
        "man": 1,
        "woman": 1,
        # UAP
        "uap": 2,
        "uap alan": 2,
        "uap alani": 2,
        "uap area": 2,
        "uap_alani": 2,
        "flying car park": 2,
        "flying_car_park": 2,
        "parking": 2,
        "park": 2,
        "landing": 2,
        "landing zone": 2,
        "ucan araba park": 2,
        "ucan araba park alani": 2,
        # UAI
        "uai": 3,
        "uai alan": 3,
        "uai alani": 3,
        "uai area": 3,
        "ambulance": 3,
        "ambulance landing": 3,
        "ambulance landing zone": 3,
        "ucan ambulans": 3,
        "ucan ambulans inis": 3,
        "ucan ambulans inis alani": 3,
        # numeric text aliases
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
    }

    @classmethod
    def valid_id_strings(cls) -> Tuple[str, ...]:
        return tuple(str(i) for i in cls.ORDERED_IDS)

    @classmethod
    def display_name(cls, class_id: int) -> str:
        return cls.DISPLAY_NAMES.get(int(class_id), "UNKNOWN")

    @classmethod
    def resolve_alias(cls, normalized_label: str) -> int:
        return cls._ALIASES.get(normalized_label, -1)

    @classmethod
    def validate_settings_contract(cls) -> None:
        configured = (
            int(Settings.CLASS_TASIT),
            int(Settings.CLASS_INSAN),
            int(Settings.CLASS_UAP),
            int(Settings.CLASS_UAI),
        )
        if configured != cls.ORDERED_IDS:
            raise DataContractError(
                "Class ID contract mismatch in Settings: "
                f"expected {cls.ORDERED_IDS}, got {configured}."
            )

    @classmethod
    def validate_model_class_order(cls, names: Dict[int, str] | List[str]) -> None:
        """Strictly validate dataset/model class order for 4-class models."""
        if isinstance(names, dict):
            sorted_items = sorted((int(k), str(v)) for k, v in names.items())
            labels = [v for _, v in sorted_items]
            indices = [k for k, _ in sorted_items]
        else:
            labels = [str(v) for v in names]
            indices = list(range(len(labels)))

        if len(labels) != len(cls.ORDERED_IDS):
            return
        if tuple(indices) != cls.ORDERED_IDS:
            raise DataContractError(
                "Model class index contract mismatch: "
                f"expected indices {cls.ORDERED_IDS}, got {tuple(indices)}."
            )

        resolved = []
        for raw in labels:
            norm = " ".join(raw.lower().replace("-", " ").replace("_", " ").split())
            resolved.append(cls.resolve_alias(norm))
        if tuple(resolved) != cls.ORDERED_IDS:
            raise DataContractError(
                "Model/dataset class order mismatch. "
                f"Expected order {cls.ORDERED_KEYS}, got labels={labels}, resolved={resolved}."
            )

    @classmethod
    def id_to_key(cls, class_id: int) -> str:
        idx = int(class_id)
        if idx not in cls.ORDERED_IDS:
            return "unknown"
        return cls.ORDERED_KEYS[idx]

