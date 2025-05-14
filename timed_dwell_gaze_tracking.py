import cv2
import numpy as np
import mediapipe as mp
from pynput.mouse import Controller
import time
import sys
import screeninfo

# ----------------------------------
# 1. Setup & shared utilities
# ----------------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mouse = Controller()

monitor = screeninfo.get_monitors()[0]
SCREEN_W, SCREEN_H = monitor.width, monitor.height

LEFT_IRIS_IDX = [473, 474, 475, 476]
RIGHT_IRIS_IDX = [468, 469, 470, 471]
LEFT_EYE_OUT = 33
RIGHT_EYE_OUT = 263
CHIN_IDX = 152
FOREHEAD_IDX = 10
NOSE_TIP_IDX = 1

CALIBRATED_POINTS = [
    (0.0, 0.0),  # top-left
    (0.5, 0.0),  # top-center
    (1.0, 0.0),  # top-right
    (0.0, 1.0),  # bottom-left
    (0.5, 1.0),  # bottom-center
    (1.0, 1.0),  # bottom-right
    (0.0, 0.5),  # mid-left
    (1.0, 0.5),  # mid-right
    (0.5, 0.5),  # center
]
DWELL_TIME = 2.0  # seconds per point


def get_avg(landmarks, idxs, w, h):
    xs = [landmarks[i].x * w for i in idxs]
    ys = [landmarks[i].y * h for i in idxs]
    return np.mean(xs), np.mean(ys)


def get_face_frame(landmarks, w, h):
    x1, y1 = landmarks[LEFT_EYE_OUT].x * w, landmarks[LEFT_EYE_OUT].y * h
    x2, y2 = landmarks[RIGHT_EYE_OUT].x * w, landmarks[RIGHT_EYE_OUT].y * h
    fw = np.hypot(x2 - x1, y2 - y1)
    xf, yf = landmarks[FOREHEAD_IDX].x * w, landmarks[FOREHEAD_IDX].y * h
    xc, yc = landmarks[CHIN_IDX].x * w, landmarks[CHIN_IDX].y * h
    fh = np.hypot(xc - xf, yc - yf)
    cx, cy = landmarks[NOSE_TIP_IDX].x * w, landmarks[NOSE_TIP_IDX].y * h
    return fw, fh, cx, cy


# ----------------------------------
# 2. Autonomous calibration
# ----------------------------------
def auto_calibrate(cap):
    src_uv = []
    dst_xy = []

    for normalized_x, normalized_y in CALIBRATED_POINTS:
        # This is the calibration point position on the full screen
        tx, ty = int(normalized_x * SCREEN_W), int(normalized_y * SCREEN_H)
        # Set fullscreen window
        cv2.namedWindow("Auto-Calibrating", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "Auto-Calibrating", cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN``
        )

        samples = []
        start = time.time()
        while time.time() - start < DWELL_TIME:
            ret, frame = cap.read()
            if not ret:
                sys.exit("Webcam error during calibration")

            # draw the calibration dot on a fullscreen black canvas
            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 40, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                f"Look at the dot ({int(DWELL_TIME - (time.time() - start))}s)",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                4,
            )
            cv2.imshow("Auto-Calibrating", canvas)
            cv2.waitKey(1)

            # This is acquiring the webcam camera feed
            # h, w, _ = frame.shape
            # # draw the target dot
            # tx, ty = int(nx * w), int(ny * h)
            # disp = frame.copy()
            # cv2.circle(disp, (tx, ty), 20, (0, 255, 0), -1)
            # cv2.putText(
            #     disp,
            #     f"Look at the dot ({int(DWELL_TIME - (time.time() - start))}s)",
            #     (30, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (255, 255, 255),
            #     2,
            # )
            # cv2.imshow("Auto-Calibrating", disp)
            # cv2.waitKey(1)

            # detect landmarks
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue

            lm = res.multi_face_landmarks[0].landmark
            lx, ly = get_avg(lm, LEFT_IRIS_IDX, w, h)
            rx, ry = get_avg(lm, RIGHT_IRIS_IDX, w, h)
            ix, iy = (lx + rx) / 2, (ly + ry) / 2
            fw, fh, cx, cy = get_face_frame(lm, w, h)
            u = (ix - cx) / fw
            v = (iy - cy) / fh
            samples.append((u, v))

        if not samples:
            sys.exit("No valid iris samples collected; please retry.")

        # median of samples for stability
        med_u, med_v = np.median(samples, axis=0)
        src_uv.append([med_u, med_v])
        dst_xy.append([nx * SCREEN_W, ny * SCREEN_H])

    cv2.destroyAllWindows()
    return np.array(src_uv), np.array(dst_xy)


# ----------------------------------
# 3. Solve affine mapping
# ----------------------------------
def solve_affine(src, dst):
    A = np.hstack([src, np.ones((src.shape[0], 1))])
    M, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)
    return M


# ----------------------------------
# 4. Real-time gaze→mouse
# ----------------------------------
def run_gaze_mouse(cap, M):
    last_x, last_y = SCREEN_W / 2, SCREEN_H / 2
    alpha = 0.3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            lx, ly = get_avg(lm, LEFT_IRIS_IDX, w, h)
            rx, ry = get_avg(lm, RIGHT_IRIS_IDX, w, h)
            ix, iy = (lx + rx) / 2, (ly + ry) / 2
            fw, fh, cx, cy = get_face_frame(lm, w, h)
            u = (ix - cx) / fw
            v = (iy - cy) / fh

            sx, sy = np.array([u, v, 1.0]) @ M
            last_x += alpha * (sx - last_x)
            last_y += alpha * (sy - last_y)
            mouse.position = (last_x, last_y)

        cv2.imshow("Gaze-Mouse", frame)
        if cv2.waitKey(1) == 27:  # Esc to exit
            break


# ----------------------------------
# 5. Main
# ----------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot open camera")

    print("Running autonomous calibration—just look at each dot.")
    src, dst = auto_calibrate(cap)
    print("Calibration complete. Solving mapping...")
    M = solve_affine(src, dst)
    print("Starting gaze-driven mouse. Press Esc in window to quit.")
    run_gaze_mouse(cap, M)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
