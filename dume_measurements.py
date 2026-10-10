
"""
DUM-E Measurement Interface
===========================

Camera-independent interface for converting image detections
into measured 3D camera-coordinate points.

Supported now:
- Store 2D object detections.
- Accept calibrated camera intrinsics.
- Convert an aligned depth-image measurement into 3D.

Not implemented:
- Reading depth from a physical depth camera.
- Estimating depth from an ordinary RGB image.
- Camera-to-robot calibration.
- Robot-arm movement.

Units:
- Camera intrinsics: pixels.
- Depth: meters after applying the specified scale.
- 3D coordinates: meters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dume_coordinates import Point3D


@dataclass(frozen=True)
class PixelDetection:
    """An object's center and bounding box in an image."""

    target_name: str
    center_x_px: float
    center_y_px: float
    confidence: float
    image_width: int
    image_height: int
    track_id: Optional[int] = None

    def __post_init__(self):
        values = (
            self.center_x_px,
            self.center_y_px,
            self.confidence,
        )

        if not all(np.isfinite(value) for value in values):
            raise ValueError("Detection values must be finite.")

        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive.")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

        if not (
            0 <= self.center_x_px < self.image_width
            and 0 <= self.center_y_px < self.image_height
        ):
            raise ValueError("Detection center lies outside the image.")


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    Pinhole camera intrinsic parameters.

    fx, fy: focal lengths in pixel units.
    cx, cy: principal point in pixel units.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int

    def __post_init__(self):
        values = (self.fx, self.fy, self.cx, self.cy)

        if not all(np.isfinite(value) for value in values):
            raise ValueError("Intrinsics must be finite.")

        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive.")

        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive.")

        if not (
            0 <= self.cx < self.image_width
            and 0 <= self.cy < self.image_height
        ):
            raise ValueError(
                "Principal point must lie within the image."
            )


@dataclass(frozen=True)
class CameraMeasurement:
    """A 2D detection and an optional measured 3D position."""

    detection: PixelDetection
    point_camera: Optional[Point3D]
    method: str
    valid_3d: bool
    reason: str

    def to_dict(self):
        return {
            "target_name": self.detection.target_name,
            "track_id": self.detection.track_id,
            "pixel_x": self.detection.center_x_px,
            "pixel_y": self.detection.center_y_px,
            "confidence": self.detection.confidence,
            "valid_3d": self.valid_3d,
            "method": self.method,
            "reason": self.reason,
            "point_camera": (
                self.point_camera.to_dict()
                if self.point_camera is not None
                else None
            ),
        }


class MeasurementProvider(ABC):
    """
    Base interface for different camera measurement methods.

    Future providers can implement this interface for depth
    cameras, stereo cameras, or calibrated RGB geometry.
    """

    @abstractmethod
    def measure(self, detection: PixelDetection) -> CameraMeasurement:
        """Produce a measurement for a detected object."""
        raise NotImplementedError


class RGBOnlyMeasurementProvider(MeasurementProvider):
    """
    Provider for an ordinary RGB camera.

    It reports image coordinates only. It intentionally does not
    invent an object's distance from a single RGB image.
    """

    def measure(self, detection: PixelDetection) -> CameraMeasurement:
        return CameraMeasurement(
            detection=detection,
            point_camera=None,
            method="rgb_2d_only",
            valid_3d=False,
            reason=(
                "No depth measurement or calibrated geometric "
                "estimate has been supplied."
            ),
        )


class AlignedDepthMeasurementProvider(MeasurementProvider):
    """
    Converts aligned depth-image samples to camera-coordinate points.

    The depth image must be aligned to the same pixel coordinates
    as the image used by YOLO-World.

    depth_scale_m_per_unit:
        Multiplier that converts raw depth values to meters.
        Example: 0.001 means each raw unit represents 1 mm.

    Invalid, zero, negative, or excessively distant depth values
    are rejected.

    Note: a center pixel can land on a hand, background, or an
    invalid depth region. A valid result is a measurement at the
    selected image location, not a guarantee of the object's
    true physical center.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        depth_scale_m_per_unit: float,
        min_depth_m: float = 0.10,
        max_depth_m: float = 5.0,
        neighborhood_radius: int = 2,
    ):
        if not np.isfinite(depth_scale_m_per_unit):
            raise ValueError("Depth scale must be finite.")

        if depth_scale_m_per_unit <= 0:
            raise ValueError("Depth scale must be positive.")

        if (
            not np.isfinite(min_depth_m)
            or not np.isfinite(max_depth_m)
            or min_depth_m <= 0
            or max_depth_m <= min_depth_m
        ):
            raise ValueError("Invalid depth limits.")

        if neighborhood_radius < 0:
            raise ValueError("Neighborhood radius cannot be negative.")

        self.intrinsics = intrinsics
        self.depth_scale = depth_scale_m_per_unit
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.neighborhood_radius = neighborhood_radius

        self.depth_image = None

    def update_depth_image(self, depth_image):
        """
        Supply the latest depth frame.

        The frame must be aligned with the detector's image,
        and its resolution must match the calibrated intrinsics.
        """
        depth = np.asarray(depth_image)

        if depth.ndim != 2:
            raise ValueError("Depth image must be a 2D array.")

        expected_shape = (
            self.intrinsics.image_height,
            self.intrinsics.image_width,
        )

        if depth.shape != expected_shape:
            raise ValueError(
                f"Expected aligned depth shape {expected_shape}; "
                f"received {depth.shape}."
            )

        if not np.issubdtype(depth.dtype, np.number):
            raise ValueError("Depth image must contain numeric values.")

        self.depth_image = depth.copy()

    def _get_valid_depth_m(self, center_x, center_y):
        if self.depth_image is None:
            return None

        u = int(round(center_x))
        v = int(round(center_y))

        radius = self.neighborhood_radius

        x_start = max(0, u - radius)
        x_end = min(
            self.intrinsics.image_width,
            u + radius + 1,
        )

        y_start = max(0, v - radius)
        y_end = min(
            self.intrinsics.image_height,
            v + radius + 1,
        )

        region = self.depth_image[y_start:y_end, x_start:x_end]

        if region.size == 0:
            return None

        values = region.astype(np.float64) * self.depth_scale

        valid = (
            np.isfinite(values)
            & (values >= self.min_depth_m)
            & (values <= self.max_depth_m)
        )

        valid_values = values[valid]

        if valid_values.size == 0:
            return None

        # Median reduces the influence of invalid or noisy samples.
        return float(np.median(valid_values))

    def measure(self, detection: PixelDetection) -> CameraMeasurement:
        if (
            detection.image_width != self.intrinsics.image_width
            or detection.image_height != self.intrinsics.image_height
        ):
            raise ValueError(
                "Detection dimensions do not match camera calibration."
            )

        depth_m = self._get_valid_depth_m(
            detection.center_x_px,
            detection.center_y_px,
        )

        if depth_m is None:
            return CameraMeasurement(
                detection=detection,
                point_camera=None,
                method="aligned_depth",
                valid_3d=False,
                reason=(
                    "No valid depth measurement was found near "
                    "the target center."
                ),
            )

        u = detection.center_x_px
        v = detection.center_y_px

        # Pinhole projection model.
        x_m = (u - self.intrinsics.cx) * depth_m / self.intrinsics.fx
        y_m = (v - self.intrinsics.cy) * depth_m / self.intrinsics.fy
        z_m = depth_m

        point = Point3D(
            x=float(x_m),
            y=float(y_m),
            z=float(z_m),
            frame="camera",
        )

        return CameraMeasurement(
            detection=detection,
            point_camera=point,
            method="aligned_depth_pinhole",
            valid_3d=True,
            reason="3D point calculated from aligned depth and intrinsics.",
        )


def main():
    print("DUM-E Measurement Interface Test")
    print("=" * 45)

    # Synthetic example calibration, for software testing only.
    intrinsics = CameraIntrinsics(
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        image_width=640,
        image_height=480,
    )

    detection = PixelDetection(
        target_name="slipper",
        center_x_px=350.0,
        center_y_px=250.0,
        confidence=0.90,
        image_width=640,
        image_height=480,
        track_id=1,
    )

    # First, demonstrate RGB-only behavior.
    rgb_provider = RGBOnlyMeasurementProvider()
    rgb_result = rgb_provider.measure(detection)

    print("\nRGB-only result:")
    print(rgb_result.to_dict())

    # Then, demonstrate the depth calculation with a synthetic
    # depth image. This is not a measurement of a real slipper.
    depth_provider = AlignedDepthMeasurementProvider(
        intrinsics=intrinsics,
        depth_scale_m_per_unit=0.001,
    )

    synthetic_depth = np.full(
        (480, 640),
        800,
        dtype=np.uint16,
    )

    depth_provider.update_depth_image(synthetic_depth)

    depth_result = depth_provider.measure(detection)

    print("\nSynthetic depth result:")
    print(depth_result.to_dict())

    print(
        "\nSynthetic values only. Real camera calibration and "
        "real depth frames are required for real measurements."
    )


if __name__ == "__main__":
    main()