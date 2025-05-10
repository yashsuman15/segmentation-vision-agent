from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        """Returns bounding box as [xmin, ymin, xmax, ymax]."""
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        """
        Create a DetectionResult from a dictionary.
        Expects detection_dict to have keys: 'score', 'label', 'box' (with xmin, ymin, xmax, ymax), and optionally 'mask'.
        """
        return cls(
            score=detection_dict['score'],
            label=detection_dict['label'],
            box=BoundingBox(
                xmin=detection_dict['box']['xmin'],
                ymin=detection_dict['box']['ymin'],
                xmax=detection_dict['box']['xmax'],
                ymax=detection_dict['box']['ymax'],
            ),
            mask=detection_dict.get('mask')
        )
