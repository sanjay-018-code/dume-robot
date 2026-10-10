
"""
DUM-E Coordinate Transformation Module

Converts a 3D point from camera coordinates to robot-base
coordinates using a calibrated homogeneous transformation.

IMPORTANT:
- Input points must be in meters.
- The transformation must be calibrated for your real setup.
- This module does not calculate depth from RGB pixels.
- Never use an example transformation to control real hardware.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Point3D:
    """A 3D point with coordinates expressed in meters."""

    x: float
    y: float
    z: float
    frame: str

    def to_dict(self):
        return {
            "x_m": self.x,
            "y_m": self.y,
            "z_m": self.z,
            "frame": self.frame,
        }


class CoordinateTransformer:
    """
    Transforms points from camera coordinates to robot-base
    coordinates.

    The matrix must describe the camera's pose relative to the
    robot base, using the convention:

        point_robot = T_robot_camera @ point_camera

    Both input and output point coordinates use meters.
    """

    def __init__(self, robot_from_camera: Sequence[Sequence[float]]):
        matrix = np.asarray(robot_from_camera, dtype=float)

        if matrix.shape != (4, 4):
            raise ValueError(
                "Transformation matrix must have shape (4, 4)."
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "Transformation matrix contains invalid numbers."
            )

        if not np.allclose(
            matrix[3],
            [0.0, 0.0, 0.0, 1.0],
            atol=1e-6,
        ):
            raise ValueError(
                "The final matrix row must be [0, 0, 0, 1]."
            )

        rotation = matrix[:3, :3]

        # A valid rigid rotation matrix must be orthonormal
        # and have determinant +1.
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-4,
        ):
            raise ValueError(
                "Rotation matrix is not orthonormal."
            )

        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=1e-4,
        ):
            raise ValueError(
                "Rotation matrix determinant must be +1."
            )

        self.robot_from_camera = matrix.copy()

    def camera_to_robot(self, point: Point3D) -> Point3D:
        """Transform one camera-frame point into robot coordinates."""

        if point.frame != "camera":
            raise ValueError(
                "Input point must use frame='camera'."
            )

        coordinates = np.array(
            [point.x, point.y, point.z, 1.0],
            dtype=float,
        )

        if not np.isfinite(coordinates).all():
            raise ValueError(
                "Point coordinates must be finite numbers."
            )

        transformed = self.robot_from_camera @ coordinates

        return Point3D(
            x=float(transformed[0]),
            y=float(transformed[1]),
            z=float(transformed[2]),
            frame="robot_base",
        )

    def robot_to_camera(self, point: Point3D) -> Point3D:
        """Transform one robot-base point into camera coordinates."""

        if point.frame != "robot_base":
            raise ValueError(
                "Input point must use frame='robot_base'."
            )

        coordinates = np.array(
            [point.x, point.y, point.z, 1.0],
            dtype=float,
        )

        if not np.isfinite(coordinates).all():
            raise ValueError(
                "Point coordinates must be finite numbers."
            )

        camera_from_robot = np.linalg.inv(
            self.robot_from_camera
        )

        transformed = camera_from_robot @ coordinates

        return Point3D(
            x=float(transformed[0]),
            y=float(transformed[1]),
            z=float(transformed[2]),
            frame="camera",
        )


def main():
    print("DUM-E Coordinate Transformation Test")
    print("-" * 45)

    # SYNTHETIC TEST ONLY — NOT A REAL CALIBRATION.
    #
    # This example represents a camera located 20 cm along
    # the robot base's X axis, with no relative rotation.
    #
    # Replace this matrix only after calibrating the real setup.

    example_robot_from_camera = [
        [1.0, 0.0, 0.0, 0.20],
        [0.0, 1.0, 0.0, 0.00],
        [0.0, 0.0, 1.0, 0.10],
        [0.0, 0.0, 0.0, 1.00],
    ]

    transformer = CoordinateTransformer(
        example_robot_from_camera
    )

    # Synthetic point 30 cm in front of the camera.
    camera_point = Point3D(
        x=0.0,
        y=0.0,
        z=0.30,
        frame="camera",
    )

    robot_point = transformer.camera_to_robot(camera_point)

    print("Synthetic camera point:")
    print(camera_point.to_dict())

    print("\nTransformed robot-base point:")
    print(robot_point.to_dict())

    print(
        "\nThis is a mathematical test, not a measurement "
        "of your physical environment."
    )

    # Check the inverse transformation.
    recovered_point = transformer.robot_to_camera(robot_point)

    print("\nPoint recovered by inverse transformation:")
    print(recovered_point.to_dict())

    assert np.allclose(
        [camera_point.x, camera_point.y, camera_point.z],
        [
            recovered_point.x,
            recovered_point.y,
            recovered_point.z,
        ],
        atol=1e-6,
    )

    print("\nPASS: Forward and inverse transformations agree.")


if __name__ == "__main__":
    main()