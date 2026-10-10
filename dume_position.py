
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TargetPosition:
    """
    Target position measured from a camera image.

    Pixel coordinates identify the target in the image.
    Normalized coordinates range approximately from -1 to +1.

    Physical 3D coordinates remain unavailable until a suitable
    calibration/depth or geometric-estimation method is added.
    """

    target_name: str
    track_id: Optional[int]

    center_x_px: float
    center_y_px: float

    normalized_x: float
    normalized_y: float

    image_width: int
    image_height: int

    box_width_px: float
    box_height_px: float

    confidence: float

    camera_x_m: Optional[float] = None
    camera_y_m: Optional[float] = None
    camera_z_m: Optional[float] = None

    def to_dict(self):
        return asdict(self)


class TargetPositionEstimator:
    """
    Estimates the target's image position and smooths its center.

    This class does not invent physical distances from pixels.
    """

    def __init__(self, smoothing_factor=0.35):
        if not 0.0 < smoothing_factor <= 1.0:
            raise ValueError(
                "smoothing_factor must be greater than 0 and at most 1"
            )

        self.smoothing_factor = smoothing_factor
        self.previous_center = None

    def reset(self):
        """Clear the previous center when changing targets."""
        self.previous_center = None

    def estimate(
        self,
        box,
        frame_shape,
        target_name,
        confidence,
        track_id=None,
    ):
        """
        Args:
            box: (x1, y1, x2, y2) bounding box in pixels.
            frame_shape: frame.shape from OpenCV.
            target_name: normalized target name.
            confidence: model confidence.
            track_id: optional ByteTrack ID.

        Returns:
            TargetPosition object.
        """
        if len(box) != 4:
            raise ValueError("box must contain x1, y1, x2, y2")

        image_height, image_width = frame_shape[:2]

        if image_width <= 0 or image_height <= 0:
            raise ValueError("Invalid image dimensions")

        x1, y1, x2, y2 = map(float, box)

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # Smooth the center to reduce frame-to-frame jitter.
        current_center = (center_x, center_y)

        if self.previous_center is None:
            smoothed_center = current_center
        else:
            alpha = self.smoothing_factor
            previous_x, previous_y = self.previous_center

            smoothed_center = (
                alpha * center_x + (1.0 - alpha) * previous_x,
                alpha * center_y + (1.0 - alpha) * previous_y,
            )

        self.previous_center = smoothed_center

        center_x, center_y = smoothed_center

        # Normalize relative to the center of the image.
        # Approximate range: -1 on the left/top, +1 right/bottom.
        normalized_x = (center_x / image_width) * 2.0 - 1.0
        normalized_y = (center_y / image_height) * 2.0 - 1.0

        return TargetPosition(
            target_name=target_name,
            track_id=track_id,
            center_x_px=round(center_x, 2),
            center_y_px=round(center_y, 2),
            normalized_x=round(normalized_x, 4),
            normalized_y=round(normalized_y, 4),
            image_width=image_width,
            image_height=image_height,
            box_width_px=round(x2 - x1, 2),
            box_height_px=round(y2 - y1, 2),
            confidence=round(float(confidence), 4),
        )