
import cv2
import torch
from ultralytics import YOLO

# Load a small pretrained object detector.
print("DUM-E: Loading vision model...")
model = YOLO("yolo11n.pt")

device = 0 if torch.cuda.is_available() else "cpu"
print("DUM-E: Using device:", device)

# Try camera 0 first. Change to 1 if needed.
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open camera. Try camera index 1."
    )

print("DUM-E: Camera active. Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        results = model.predict(
            source=frame,
            device=device,
            imgsz=640,
            conf=0.35,
            verbose=False
        )

        # Draw detected objects and their bounding boxes.
        annotated_frame = results[0].plot()

        cv2.imshow("DUM-E Vision", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("DUM-E: Vision stopped.")