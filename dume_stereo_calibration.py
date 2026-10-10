
import cv2
import numpy as np
import os
import time

# ============================================================
# DUM-E STEREO CAMERA CALIBRATION
# ============================================================

LEFT_CAMERA_ID = 0
RIGHT_CAMERA_ID = 1

# Number of INNER corners in the printed checkerboard
CHECKERBOARD = (9, 6)

# Measure one checkerboard square in millimeters.
# Change this if your printed square size is different.
SQUARE_SIZE_MM = 25.0

IMAGE_DIR = "stereo_calibration_images"
OUTPUT_FILE = "dume_stereo_calibration.npz"

MINIMUM_PAIRS = 15


def create_object_points():
    points = np.zeros(
        (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
        np.float32,
    )

    points[:, :2] = np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1],
    ].T.reshape(-1, 2)

    points *= SQUARE_SIZE_MM
    return points


def open_cameras():
    left = cv2.VideoCapture(LEFT_CAMERA_ID)
    right = cv2.VideoCapture(RIGHT_CAMERA_ID)

    if not left.isOpened():
        left.release()
        right.release()
        raise RuntimeError(
            f"Could not open left camera {LEFT_CAMERA_ID}"
        )

    if not right.isOpened():
        left.release()
        right.release()
        raise RuntimeError(
            f"Could not open right camera {RIGHT_CAMERA_ID}. "
            "Check the camera ID or connect the second webcam."
        )

    return left, right


def capture_calibration_images():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    left_dir = os.path.join(IMAGE_DIR, "left")
    right_dir = os.path.join(IMAGE_DIR, "right")

    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    left_camera, right_camera = open_cameras()
    pair_count = 0

    print("\nSTEREO CALIBRATION IMAGE CAPTURE")
    print("--------------------------------")
    print("Show the same checkerboard to BOTH cameras.")
    print("Move it to different positions and angles.")
    print("C = capture pair | Q = finish\n")

    try:
        while True:
            left_ok, left_frame = left_camera.read()
            right_ok, right_frame = right_camera.read()

            if not left_ok or not right_ok:
                print("Camera frame capture failed.")
                break

            left_display = left_frame.copy()
            right_display = right_frame.copy()

            left_gray = cv2.cvtColor(
                left_frame, cv2.COLOR_BGR2GRAY
            )
            right_gray = cv2.cvtColor(
                right_frame, cv2.COLOR_BGR2GRAY
            )

            left_found, left_corners = cv2.findChessboardCorners(
                left_gray,
                CHECKERBOARD,
                None,
            )

            right_found, right_corners = cv2.findChessboardCorners(
                right_gray,
                CHECKERBOARD,
                None,
            )

            if left_found:
                cv2.drawChessboardCorners(
                    left_display,
                    CHECKERBOARD,
                    left_corners,
                    left_found,
                )

            if right_found:
                cv2.drawChessboardCorners(
                    right_display,
                    CHECKERBOARD,
                    right_corners,
                    right_found,
                )

            cv2.putText(
                left_display,
                f"Left checkerboard: {left_found}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if left_found else (0, 0, 255),
                2,
            )

            cv2.putText(
                right_display,
                f"Right checkerboard: {right_found}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if right_found else (0, 0, 255),
                2,
            )

            cv2.imshow("DUM-E Left Camera", left_display)
            cv2.imshow("DUM-E Right Camera", right_display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                if not (left_found and right_found):
                    print(
                        "Checkerboard must be visible in both cameras."
                    )
                    continue

                left_path = os.path.join(
                    left_dir, f"left_{pair_count:03d}.png"
                )
                right_path = os.path.join(
                    right_dir, f"right_{pair_count:03d}.png"
                )

                cv2.imwrite(left_path, left_frame)
                cv2.imwrite(right_path, right_frame)

                pair_count += 1
                print(f"Saved calibration pair {pair_count}")

                time.sleep(0.2)

            elif key == ord("q"):
                break

    finally:
        left_camera.release()
        right_camera.release()
        cv2.destroyAllWindows()

    print(f"\nCaptured {pair_count} stereo image pairs.")
    return pair_count


def calibrate_stereo():
    left_dir = os.path.join(IMAGE_DIR, "left")
    right_dir = os.path.join(IMAGE_DIR, "right")

    left_files = sorted(
        f for f in os.listdir(left_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    right_files = sorted(
        f for f in os.listdir(right_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    if len(left_files) != len(right_files):
        raise RuntimeError(
            "Left and right image counts do not match."
        )

    if len(left_files) < MINIMUM_PAIRS:
        raise RuntimeError(
            f"Need at least {MINIMUM_PAIRS} image pairs. "
            f"Found {len(left_files)}."
        )

    object_points = []
    left_image_points = []
    right_image_points = []

    image_size = None
    template_points = create_object_points()

    print("\nFinding checkerboards in saved images...")

    for left_name, right_name in zip(left_files, right_files):
        left_path = os.path.join(left_dir, left_name)
        right_path = os.path.join(right_dir, right_name)

        left_image = cv2.imread(left_path)
        right_image = cv2.imread(right_path)

        if left_image is None or right_image is None:
            print(f"Skipping unreadable pair: {left_name}")
            continue

        if left_image.shape[:2] != right_image.shape[:2]:
            print(f"Skipping different-sized pair: {left_name}")
            continue

        gray_left = cv2.cvtColor(
            left_image, cv2.COLOR_BGR2GRAY
        )
        gray_right = cv2.cvtColor(
            right_image, cv2.COLOR_BGR2GRAY
        )

        image_size = (
            gray_left.shape[1],
            gray_left.shape[0],
        )

        found_left, corners_left = cv2.findChessboardCorners(
            gray_left, CHECKERBOARD, None
        )

        found_right, corners_right = cv2.findChessboardCorners(
            gray_right, CHECKERBOARD, None
        )

        if not (found_left and found_right):
            print(f"Skipping pair without both boards: {left_name}")
            continue

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        corners_left = cv2.cornerSubPix(
            gray_left,
            corners_left,
            (11, 11),
            (-1, -1),
            criteria,
        )

        corners_right = cv2.cornerSubPix(
            gray_right,
            corners_right,
            (11, 11),
            (-1, -1),
            criteria,
        )

        object_points.append(template_points.copy())
        left_image_points.append(corners_left)
        right_image_points.append(corners_right)

    if len(object_points) < MINIMUM_PAIRS:
        raise RuntimeError(
            f"Only {len(object_points)} usable pairs. "
            f"Need at least {MINIMUM_PAIRS}. "
            "Capture more varied checkerboard views."
        )

    print(f"Calibrating using {len(object_points)} image pairs...")

    flags = 0
    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )

    rms_left, K1, D1, _, _ = cv2.calibrateCamera(
        object_points,
        left_image_points,
        image_size,
        None,
        None,
    )

    rms_right, K2, D2, _, _ = cv2.calibrateCamera(
        object_points,
        right_image_points,
        image_size,
        None,
        None,
    )

    stereo_rms, K1, D1, K2, D2, R, T, E, F = (
        cv2.stereoCalibrate(
            object_points,
            left_image_points,
            right_image_points,
            K1,
            D1,
            K2,
            D2,
            image_size,
            criteria=criteria,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
    )

    R1, R2, P1, P2, Q, roi1, roi2 = (
        cv2.stereoRectify(
            K1,
            D1,
            K2,
            D2,
            image_size,
            R,
            T,
            alpha=0,
        )
    )

    baseline_mm = float(np.linalg.norm(T))

    np.savez_compressed(
        OUTPUT_FILE,
        image_size=np.array(image_size),
        checkerboard=np.array(CHECKERBOARD),
        square_size_mm=np.array(SQUARE_SIZE_MM),
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi1=np.array(roi1),
        roi2=np.array(roi2),
        left_rms=np.array(rms_left),
        right_rms=np.array(rms_right),
        stereo_rms=np.array(stereo_rms),
        baseline_mm=np.array(baseline_mm),
    )

    print("\nCALIBRATION COMPLETE")
    print("--------------------")
    print(f"Left camera RMS:  {rms_left:.4f}")
    print(f"Right camera RMS: {rms_right:.4f}")
    print(f"Stereo RMS:       {stereo_rms:.4f}")
    print(f"Camera baseline:  {baseline_mm:.2f} mm")
    print(f"Saved file:       {OUTPUT_FILE}")
    print(
        "\nCalibration is not yet proof of accurate depth. "
        "Test measurements against objects at known distances."
    )


def main():
    print("=" * 55)
    print("DUM-E STEREO CALIBRATION")
    print("=" * 55)

    try:
        if not os.path.isdir(IMAGE_DIR):
            count = capture_calibration_images()

            if count < MINIMUM_PAIRS:
                print(
                    "\nNot enough image pairs yet. Run this program "
                    "again to capture more pairs."
                )
                return

        calibrate_stereo()

    except Exception as exc:
        print(f"\nCalibration error: {exc}")
        print(
            "\nCheck camera IDs, checkerboard dimensions, "
            "square size, and saved calibration images."
        )


if __name__ == "__main__":
    main()