
import cv2
import torch
import requests
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3:latest"
YOLO_MODEL = "yolo11n.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.35

# Map common descriptions to YOLO's trained categories.
ALIASES = {
    "water bottle": "bottle",
    "plastic bottle": "bottle",
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "smartphone": "cell phone",
    "television remote": "remote",
    "tv remote": "remote",
    "sofa": "couch",
    "aeroplane": "airplane",
    "motorbike": "motorcycle",
}

# ---------------- LANGUAGE PLANNER ----------------

SYSTEM_PROMPT = """
Convert the user's instruction into JSON with these fields:
- goal: short description
- target_object: the main requested object, or null
- action: "find", "retrieve", "describe", or "other"
- needs_clarification: true or false
- response: a short explanation

Return JSON only.
Do not claim that an object has been found.
Do not claim that a robot has moved or picked anything up.
Extract the object the user actually wants.
"""


def plan_command(command):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": command},
            ],
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


# ---------------- OBJECT MATCHING ----------------

def normalize_name(name):
    name = name.lower().strip()

    # Handle common object descriptions.
    name = ALIASES.get(name, name)

    # Basic plural handling.
    if name.endswith("s") and name[:-1] in YOLO_LABELS:
        name = name[:-1]

    return name


# ---------------- CAMERA SEARCH ----------------

def search_for_object(target, model, camera):
    labels = model.names
    global YOLO_LABELS
    YOLO_LABELS = set(labels.values())

    target = normalize_name(target)

    if target not in YOLO_LABELS:
        print(f"\nDUM-E: My current detector does not have")
        print(f"a trained category for '{target}'.")
        print("I cannot reliably search for that object yet.")
        print("We will need open-vocabulary vision or a custom detector.")
        return

    print(f"\nDUM-E: Searching the camera view for '{target}'.")
    print("Move the object into view if necessary.")
    print("Press Q in the camera window to stop searching.")

    frame_count = 0

    while True:
        ok, frame = camera.read()

        if not ok:
            print("DUM-E: Could not read a camera frame.")
            break

        results = model.predict(
            source=frame,
            device=0 if torch.cuda.is_available() else "cpu",
            imgsz=640,
            conf=CONFIDENCE,
            verbose=False,
        )

        result = results[0]
        found = False

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detected_name = labels[class_id]
            confidence = float(box.conf[0].item())

            if detected_name == target:
                found = True

                x1, y1, x2, y2 = (
                    int(value) for value in box.xyxy[0].tolist()
                )

                height, width = frame.shape[:2]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Relative position in the camera image.
                horizontal = (
                    "left" if center_x < width / 3
                    else "right" if center_x > 2 * width / 3
                    else "center"
                )

                vertical = (
                    "upper" if center_y < height / 3
                    else "lower" if center_y > 2 * height / 3
                    else "middle"
                )

                print(
                    f"\nDUM-E: Found '{target}' "
                    f"(confidence {confidence:.2f})."
                )
                print(
                    f"Image position: {horizontal}, {vertical}"
                )
                print(
                    f"Bounding-box center: "
                    f"x={center_x}, y={center_y}"
                )
                print(
                    "This is an image position, not a physical "
                    "distance or robot-arm coordinate."
                )
                break

        annotated = result.plot()

        if found:
            cv2.putText(
                annotated,
                f"FOUND: {target}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                annotated,
                f"SEARCHING: {target}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 165, 255),
                2,
            )

        cv2.imshow("DUM-E Object Search", annotated)

        frame_count += 1

        if not found and frame_count % 60 == 0:
            print(
                f"DUM-E: '{target}' is not visible "
                "in the current camera frames."
            )

        # Keep the found object visible until Q is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Stop after finding the requested object.
        if found:
            print("DUM-E: Search complete.")
            break


def main():
    print("DUM-E: Initializing language and vision systems...")

    # Confirm Ollama is reachable before opening the camera.
    try:
        requests.get(
            "http://localhost:11434/api/tags",
            timeout=5,
        ).raise_for_status()
    except requests.RequestException as error:
        print("Ollama is unavailable:", error)
        return

    model = YOLO(YOLO_MODEL)
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Could not open camera index 0.")
        print("Try changing CAMERA_INDEX to 1.")
        return

    try:
        print("\nDUM-E is ready.")
        print("Examples: Find a bottle. Find my charger.")
        print("Type 'exit' to quit.")

        while True:
            command = input("\nYou: ").strip()

            if command.lower() == "exit":
                break

            if not command:
                continue

            try:
                import json
                plan = json.loads(plan_command(command))

                print("\nDUM-E's interpretation:")
                print(json.dumps(plan, indent=2))

                target = plan.get("target_object")
                action = plan.get("action")
                needs_clarification = plan.get(
                    "needs_clarification", False
                )

                if needs_clarification or not target:
                    print(
                        "DUM-E: Please clarify which object "
                        "you want me to find."
                    )
                    continue

                if action not in ("find", "retrieve"):
                    print(
                        "DUM-E: This demo only searches for "
                        "objects; no physical action is performed."
                    )
                    continue

                search_for_object(target, model, camera)

            except requests.RequestException as error:
                print("Ollama request failed:", error)
            except (ValueError, KeyError) as error:
                print("Could not interpret the planner response:", error)

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("DUM-E: Systems shut down.")


if __name__ == "__main__":
    main()