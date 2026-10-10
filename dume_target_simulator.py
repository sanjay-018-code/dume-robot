"""
DUM-E Step 7: Safe Target-Planning Simulation

Purpose:
1. Accept measurement dictionaries from dume_measurements.py.
2. Reject measurements that do not contain valid 3D coordinates.
3. Transform camera coordinates into a hypothetical robot-base frame.
4. Check the proposed point against a demonstration workspace.
5. Print a proposed target without moving any hardware.

IMPORTANT:
All example coordinates and the transformation matrix are synthetic.
Do not use this demo matrix or workspace limits to control a real robot.
"""

from dataclasses import dataclass, asdict
from typing import Optional

from dume_coordinates import CoordinateTransformer, Point3D


@dataclass
class WorkspaceBounds:
    """Demonstration-only bounds in robot-base coordinates, in metres."""
    x_min: float = -0.50
    x_max: float = 0.50
    y_min: float = -0.50
    y_max: float = 0.50
    z_min: float = 0.00
    z_max: float = 0.80

    def contains(self, point: Point3D) -> bool:
        return (
            self.x_min <= point.x <= self.x_max
            and self.y_min <= point.y <= self.y_max
            and self.z_min <= point.z <= self.z_max
        )


@dataclass
class TargetPlan:
    target_name: str
    accepted: bool
    status: str
    reason: str
    camera_point: Optional[dict] = None
    robot_base_point: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


class TargetPlanner:
    """
    Plans a target point from a measurement dictionary.

    Expected dictionary format:
    {
        "target_name": "slipper",
        "valid_3d": True,
        "point_camera": {
            "x_m": 0.04,
            "y_m": 0.0133,
            "z_m": 0.8,
            "frame": "camera"
        }
    }
    """

    def __init__(
        self,
        transformer: CoordinateTransformer,
        workspace: Optional[WorkspaceBounds] = None,
    ):
        self.transformer = transformer
        self.workspace = workspace or WorkspaceBounds()

    def plan(self, measurement: dict) -> TargetPlan:
        target_name = str(measurement.get("target_name", "unknown"))
        point_data = measurement.get("point_camera")

        # Never infer 3D coordinates from a 2D image detection.
        if measurement.get("valid_3d") is not True:
            return TargetPlan(
                target_name=target_name,
                accepted=False,
                status="REJECTED",
                reason=(
                    "Measurement has no valid 3D position. "
                    "RGB-only detections cannot provide real-world depth."
                ),
            )

        if not isinstance(point_data, dict):
            return TargetPlan(
                target_name=target_name,
                accepted=False,
                status="REJECTED",
                reason="The 3D camera point is missing or malformed.",
            )

        try:
            x = float(point_data["x_m"])
            y = float(point_data["y_m"])
            z = float(point_data["z_m"])
            frame = point_data.get("frame", "camera")
        except (KeyError, TypeError, ValueError):
            return TargetPlan(
                target_name=target_name,
                accepted=False,
                status="REJECTED",
                reason="The 3D point must contain numeric x_m, y_m and z_m values.",
            )

        if frame != "camera":
            return TargetPlan(
                target_name=target_name,
                accepted=False,
                status="REJECTED",
                reason=f"Expected camera-frame coordinates, received {frame!r}.",
            )

        if not all(map(self._is_finite, (x, y, z))):
            return TargetPlan(
                target_name=target_name,
                accepted=False,
                status="REJECTED",
                reason="The 3D point contains a non-finite coordinate.",
            )

        if z <= 0:
            return TargetPlan(
                target_name=target_name,
                accepted=False,
                status="REJECTED",
                reason="The target depth must be positive.",
            )

        camera_point = Point3D(x=x, y=y, z=z, frame="camera")

        try:
            robot_point = self.transformer.camera_to_robot(camera_point)
        except (ValueError, TypeError, AttributeError) as exc:
            return TargetPlan(
                target_name=target_name,
                accepted=False,
                status="REJECTED",
                reason=f"Coordinate transformation failed: {exc}",
            )

        camera_dict = {
            "x_m": round(camera_point.x, 4),
            "y_m": round(camera_point.y, 4),
            "z_m": round(camera_point.z, 4),
            "frame": "camera",
        }

        robot_dict = {
            "x_m": round(robot_point.x, 4),
            "y_m": round(robot_point.y, 4),
            "z_m": round(robot_point.z, 4),
            "frame": robot_point.frame,
        }

        if not self.workspace.contains(robot_point):
            return TargetPlan(
                target_name=target_name,
                accepted=False,
                status="REJECTED",
                reason=(
                    "The proposed point is outside the demonstration workspace. "
                    "No movement was requested."
                ),
                camera_point=camera_dict,
                robot_base_point=robot_dict,
            )

        return TargetPlan(
            target_name=target_name,
            accepted=True,
            status="SIMULATED_ONLY",
            reason=(
                "3D point transformed and passed the demonstration workspace check. "
                "This is a proposed target only; no hardware was moved."
            ),
            camera_point=camera_dict,
            robot_base_point=robot_dict,
        )

    @staticmethod
    def _is_finite(value: float) -> bool:
        import math
        return math.isfinite(value)


def print_plan(title: str, plan: TargetPlan) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Target:   {plan.target_name}")
    print(f"Accepted: {plan.accepted}")
    print(f"Status:   {plan.status}")
    print(f"Reason:   {plan.reason}")

    if plan.camera_point is not None:
        print(f"Camera point:     {plan.camera_point}")

    if plan.robot_base_point is not None:
        print(f"Robot-base point: {plan.robot_base_point}")


def main() -> None:
    print("DUM-E Target Planner — Simulation Only")
    print("=" * 45)

    # SYNTHETIC transform for software testing only.
    # This is NOT a measured camera-to-robot calibration.
    synthetic_transform = [
        [1.0, 0.0, 0.0, 0.20],
        [0.0, 1.0, 0.0, 0.00],
        [0.0, 0.0, 1.0, 0.10],
        [0.0, 0.0, 0.0, 1.00],
    ]

    transformer = CoordinateTransformer(
        robot_from_camera=synthetic_transform
    )

    planner = TargetPlanner(
        transformer=transformer,
        workspace=WorkspaceBounds(),
    )

    # Example 1: matches the RGB-only result from dume_measurements.py.
    rgb_only_measurement = {
        "target_name": "slipper",
        "track_id": 1,
        "pixel_x": 350.0,
        "pixel_y": 250.0,
        "confidence": 0.9,
        "valid_3d": False,
        "method": "rgb_2d_only",
        "reason": "No depth measurement is available.",
        "point_camera": None,
    }

    rgb_plan = planner.plan(rgb_only_measurement)
    print_plan("Test 1: RGB-only detection", rgb_plan)

    # Example 2: mirrors the synthetic depth output from Step 6.
    synthetic_depth_measurement = {
        "target_name": "slipper",
        "track_id": 1,
        "pixel_x": 350.0,
        "pixel_y": 250.0,
        "confidence": 0.9,
        "valid_3d": True,
        "method": "aligned_depth_pinhole",
        "reason": "Synthetic depth and intrinsics were used.",
        "point_camera": {
            "x_m": 0.04,
            "y_m": 0.013333333333333334,
            "z_m": 0.8,
            "frame": "camera",
        },
    }

    depth_plan = planner.plan(synthetic_depth_measurement)
    print_plan("Test 2: Synthetic depth measurement", depth_plan)

    print("\n" + "=" * 45)
    print("SIMULATION COMPLETE — NO HARDWARE MOVEMENT")
    print("All coordinates and calibration values are synthetic.")


if __name__ == "__main__":
    main()