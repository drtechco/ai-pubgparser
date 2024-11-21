import json
import base64


class LabelMe:
    def __init__(self, width: int, height: int, filename: str):
        """
        new labelme object
        """
        self.label = {
            "version": "5.4.1",
            "flags": {},
            "shapes": [],
            "imagePath": str(filename),
            "imageData": None,
            "imageHeight": int(height),
            "imageWidth": int(width),
        }

    def add_label(self, label: str, points: list, shape_type: str, conf: float) -> None:
        assert len(points) == 2, "points length must exactly be two"
        _label = {
            "label": label,
            "points": points,
            "group_id": f"{conf}",
            "description": "",
            "shape_type": shape_type,
            "mask": None,
        }
        self.label["shapes"].append(_label)

    def to_dict(self):
        return self.label

    def __repr__(self):
        return str(self.label)
