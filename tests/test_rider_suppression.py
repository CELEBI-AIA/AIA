import unittest

from config.settings import Settings
try:
    from src.detection import ObjectDetector
except Exception:  # pragma: no cover - environment-dependent
    ObjectDetector = None


@unittest.skipUnless(ObjectDetector is not None, "detection runtime deps are missing")
class TestRiderSuppression(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "RIDER_SUPPRESS_ENABLED": Settings.RIDER_SUPPRESS_ENABLED,
            "RIDER_OVERLAP_THRESHOLD": Settings.RIDER_OVERLAP_THRESHOLD,
            "RIDER_IOU_THRESHOLD": Settings.RIDER_IOU_THRESHOLD,
            "RIDER_SOURCE_CLASSES": Settings.RIDER_SOURCE_CLASSES,
        }
        Settings.RIDER_SUPPRESS_ENABLED = True
        Settings.RIDER_OVERLAP_THRESHOLD = 0.35
        Settings.RIDER_IOU_THRESHOLD = 0.15
        Settings.RIDER_SOURCE_CLASSES = (1, 3, 10)

    def tearDown(self):
        for key, value in self._orig.items():
            setattr(Settings, key, value)

    @staticmethod
    def _det(cls_int, source_cls_id, bbox):
        x1, y1, x2, y2 = bbox
        return {
            "cls_int": cls_int,
            "cls": str(cls_int),
            "source_cls_id": source_cls_id,
            "bbox": (x1, y1, x2, y2),
            "top_left_x": x1,
            "top_left_y": y1,
            "bottom_right_x": x2,
            "bottom_right_y": y2,
            "confidence": 0.9,
        }

    def test_person_over_bicycle_is_suppressed(self):
        person = self._det(1, 0, (100, 100, 140, 160))
        bicycle = self._det(0, 1, (95, 105, 145, 165))

        out = ObjectDetector._suppress_rider_persons([person, bicycle])

        self.assertEqual(sum(1 for d in out if d["cls_int"] == 1), 0)
        self.assertEqual(sum(1 for d in out if d["cls_int"] == 0), 1)

    def test_person_over_car_not_suppressed(self):
        person = self._det(1, 0, (100, 100, 140, 160))
        car = self._det(0, 2, (95, 105, 145, 165))  # car source id

        out = ObjectDetector._suppress_rider_persons([person, car])

        self.assertEqual(sum(1 for d in out if d["cls_int"] == 1), 1)

    def test_low_overlap_not_suppressed(self):
        person = self._det(1, 0, (100, 100, 140, 160))
        bike = self._det(0, 1, (200, 200, 240, 260))

        out = ObjectDetector._suppress_rider_persons([person, bike])

        self.assertEqual(sum(1 for d in out if d["cls_int"] == 1), 1)

    def test_multi_person_only_overlapping_suppressed(self):
        rider = self._det(1, 0, (100, 100, 140, 160))
        walker = self._det(1, 0, (220, 120, 260, 180))
        moto = self._det(0, 3, (95, 105, 145, 165))

        out = ObjectDetector._suppress_rider_persons([rider, walker, moto])

        self.assertEqual(sum(1 for d in out if d["cls_int"] == 1), 1)
        self.assertEqual(sum(1 for d in out if d["cls_int"] == 0), 1)


if __name__ == "__main__":
    unittest.main()
