
import time

import cv2
import torch
from ultralytics import YOLOWorld

from dume_camera import CameraSource
from dume_position import TargetPositionEstimator


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolov8s-worldv2.pt"
CAMERA_SOURCE = 0

CONFIDENCE_THRESHOLD = 0.10
IMAGE_SIZE = 640

CONFIRMATION_FRAMES = 4
LOST_FRAME_LIMIT = 15

WINDOW_NAME = "DUM-E | Target Position Estimation"


OBJECT_PROMPTS = {
    "slipper": [
        "slipper",
        "flip-flop",
        "rubber sandal",
        "open-toe sandal",
        "rubber footwear",
        "slipper viewed from above",
        "slipper sole",
    ],
    "shoe": ["shoe", "sneaker", "footwear"],
    "phone": ["mobile phone", "cell phone", "smartphone"],
    "charger": [
        "phone charger",
        "charging adapter",
        "USB charger",
        "power adapter",
    ],
    "bottle": [
        "water bottle",
        "plastic bottle",
        "drinking bottle",
    ],
}

ALIASES = {
    "cheppal": "slipper",
    "chappal": "slipper",
    "sandal": "slipper",
    "flipflop": "slipper",
    "flip-flop": "slipper",
    "mobile": "phone",
    "mobile phone": "phone",
    "cell phone": "phone",
    "smartphone": "phone",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_object_name(name):
    name = name.strip().lower()
    return ALIASES.get(name, name)


def get_target_prompts(name):
    normalized = normalize_object_name(name)
    return normalized, OBJECT_PROMPTS.get(normalized, [normalized])


def get_target_from_command():
    command = input(
        "Enter a command (example: find cheppal): "
    ).strip()

    if not command:
        command = "find slipper"

    target_name = None

    try:
        from dume_planner import plan_command

        plan = plan_command(command)

        if isinstance(plan, dict):
            target_name = plan.get("target_object")

    except Exception as exc:
        print(f"Planner unavailable: {exc}")

    if not target_name:
        words = command.lower().split()

        if words and words[0] in {
            "find", "locate", "search", "track"
        }:
            words = words[1:]

        target_name = " ".join(words).strip()

    return target_name or "slipper"


def draw_text(frame, text, position, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("DUM-E — CAMERA-INDEPENDENT POSITION ESTIMATION")
    print("=" * 60)

    if torch.cuda.is_available():
        device = 0
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("CUDA unavailable; using CPU.")

    requested_target = get_target_from_command()
    target_name, prompts = get_target_prompts(requested_target)

    print("Target:", target_name)
    print("Prompts:", prompts)

    print("\nLoading YOLO-World...")
    model = YOLOWorld(MODEL_PATH)
    model.set_classes(prompts)

    # Dedicated position-estimation component.
    position_estimator = TargetPositionEstimator(
        smoothing_factor=0.35
    )

    camera = None

    selected_track_id = None
    lost_frames = 0
    consecutive_frames = 0

    last_position = None
    last_box = None

    fps = 0.0
    previous_time = time.perf_counter()

    try:
        camera = CameraSource(CAMERA_SOURCE)

        print("\nCamera ready.")
        print("Press Q to quit.")
        print("Note: displayed coordinates are image coordinates,")
        print("not physical distances in meters.\n")

        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print("Unable to read camera frame.")
                break

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                device=device,
                verbose=False,
            )

            result = results[0]
            detections = []

            if result.boxes is not None:
                boxes = result.boxes

                for index in range(len(boxes)):
                    confidence = float(
                        boxes.conf[index].item()
                    )
                    class_id = int(
                        boxes.cls[index].item()
                    )

                    coords = (
                        boxes.xyxy[index]
                        .detach()
                        .cpu()
                        .tolist()
                    )

                    x1, y1, x2, y2 = map(int, coords)

                    track_id = None

                    if boxes.id is not None:
                        track_id = int(
                            boxes.id[index].item()
                        )

                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0

                    detections.append({
                        "id": track_id,
                        "confidence": confidence,
                        "class_name": result.names.get(
                            class_id, target_name
                        ),
                        "box": (x1, y1, x2, y2),
                        "center": (center_x, center_y),
                    })

            chosen = None

            # First, preserve the existing tracker identity.
            if selected_track_id is not None:
                for detection in detections:
                    if detection["id"] == selected_track_id:
                        chosen = detection
                        break

            # If the target is missing, look for a nearby candidate.
            if chosen is None and detections:
                if last_position is not None and lost_frames <= LOST_FRAME_LIMIT:
                    old_x = last_position.center_x_px
                    old_y = last_position.center_y_px

                    frame_height, frame_width = frame.shape[:2]

                    max_distance = (
                        frame_width ** 2 + frame_height ** 2
                    ) ** 0.5 * 0.25

                    nearby = []

                    for detection in detections:
                        new_x, new_y = detection["center"]

                        distance = (
                            (new_x - old_x) ** 2
                            + (new_y - old_y) ** 2
                        ) ** 0.5

                        if distance <= max_distance:
                            nearby.append((distance, detection))

                    if nearby:
                        nearby.sort(key=lambda item: item[0])
                        chosen = nearby[0][1]

                if chosen is None:
                    chosen = max(
                        detections,
                        key=lambda item: item["confidence"],
                    )

            status = "SEARCHING"

            if chosen is not None:
                lost_frames = 0
                consecutive_frames += 1

                if chosen["id"] is not None:
                    selected_track_id = chosen["id"]

                last_box = chosen["box"]

                last_position = position_estimator.estimate(
                    box=chosen["box"],
                    frame_shape=frame.shape,
                    target_name=target_name,
                    confidence=chosen["confidence"],
                    track_id=chosen["id"],
                )

                if consecutive_frames >= CONFIRMATION_FRAMES:
                    status = "TRACKING"
                else:
                    status = "CONFIRMING TARGET"

                x1, y1, x2, y2 = chosen["box"]

                center_x = int(last_position.center_x_px)
                center_y = int(last_position.center_y_px)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    6,
                    (0, 0, 255),
                    -1,
                )

                draw_text(
                    frame,
                    f"{chosen['class_name']} "
                    f"ID:{chosen['id']} "
                    f"Conf:{chosen['confidence']:.2f}",
                    (x1, max(22, y1 - 10)),
                    (0, 255, 0),
                )

                draw_text(
                    frame,
                    f"Pixel position: ({center_x}, {center_y})",
                    (15, 125),
                )

                draw_text(
                    frame,
                    "Normalized image position: "
                    f"({last_position.normalized_x:.3f}, "
                    f"{last_position.normalized_y:.3f})",
                    (15, 155),
                )

                # Print structured position data periodically
                # in the console only when a target is confirmed.
                if status == "TRACKING":
                    print(
                        "\r"
                        f"Target={target_name} | "
                        f"ID={selected_track_id} | "
                        f"Pixel=({center_x}, {center_y}) | "
                        f"Normalized=("
                        f"{last_position.normalized_x:.3f}, "
                        f"{last_position.normalized_y:.3f}) | "
                        f"Confidence={last_position.confidence:.2f}",
                        end="",
                        flush=True,
                    )

            else:
                consecutive_frames = 0
                lost_frames += 1

                if lost_frames <= LOST_FRAME_LIMIT:
                    status = "TARGET TEMPORARILY LOST"
                else:
                    status = "SEARCHING / REACQUIRING"

                    selected_track_id = None
                    last_box = None
                    last_position = None
                    position_estimator.reset()

            # FPS calculation.
            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            if elapsed > 0:
                current_fps = 1.0 / elapsed

                fps = (
                    current_fps
                    if fps == 0
                    else 0.9 * fps + 0.1 * current_fps
                )

            draw_text(
                frame,
                f"Target: {target_name}",
                (15, 30),
            )
            draw_text(
                frame,
                f"Status: {status}",
                (15, 60),
                (0, 255, 255),
            )
            draw_text(
                frame,
                f"Tracker ID: {selected_track_id}",
                (15, 90),
            )
            draw_text(
                frame,
                f"FPS: {fps:.1f}",
                (15, 185),
            )

            if last_position is not None:
                draw_text(
                    frame,
                    "3D position: NOT AVAILABLE",
                    (15, 215),
                    (0, 165, 255),
                )

            draw_text(
                frame,
                "Q: Quit",
                (15, frame.shape[0] - 15),
            )

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopping DUM-E...")

    finally:
        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()
        print("\nDUM-E vision system stopped.")


if __name__ == "__main__":
    main()