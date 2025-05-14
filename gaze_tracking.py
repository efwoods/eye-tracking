import cv2
import numpy as np
import mediapipe as mp
from pynput.mouse import Controller
import time
import sys
import screeninfo


# ------------------
# 1. Setup
# ------------------

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mouse = Controller()

# Acquiring the actual screen display size
monitor = screeninfo.get_monitors()[0]
SCREEN_W, SCREEN_H = monitor.width, monitor.height

# Eye landmarks indices for iris center (MediaPipe FaceMesh)
# Left iris center is average of landmarks 473, 474, 475, 476
LEFT_IRIS_IDX = [473, 474, 475, 476]
RIGHT_IRIS_IDX = [468, 469, 470, 471]

CAL_POINTS_NORM = [
    (0.1, 0.1),  # top-left
    (0.9, 0.1),  # top-right
    (0.1, 0.9),  # bottom-left
    (0.9, 0.9),  # bottom-right
    (0.5, 0.5),  # center
]


def get_iris_center(landmarks, idx_list, img_w, img_h):
    xs = [landmarks[i].x * img_w for i in idx_list]
    ys = [landmarks[i].y * img_h for i in idx_list]
    return np.mean(xs), np.mean(ys)


# --------------
# 2. Calibration
# --------------
def calibrate(cap):
    cal_src = []  # [ [iris_x, iris_y], ... ]
    cal_dst = []  # [ [screen_x, screen_y], ... ]
    for nx, ny in CAL_POINTS_NORM:
        # Draw fulll-screen point
        while True:
            ret, frame = cap.read()
            if not ret:
                sys.exit("Webcam not available")
            h, w, _ = frame.shape
            # draw calibration point
            px, py = int(nx * w), int(ny * h)
            disp = frame.copy()
            cv2.circle(disp, (px, py), 20, (0, 255, 0), -1)
            cv2.putText(
                disp,
                "Press Space to record here",
                (30, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Calibration", disp)
            key = cv2.waitKey(1)
            if key == 32:  # space bar
                # capture a frame, detect face
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if not res.multi_face_landmarks:
                    continue  # try again until face is found
                lm = res.multi_face_landmarks[0].landmark
                # average both eyes
                lx, ly = get_iris_center(lm, LEFT_IRIS_IDX, w, h)
                rx, ry = get_iris_center(lm, RIGHT_IRIS_IDX, w, h)
                ix, iy = (lx + rx) / 2, (ly + ry) / 2
                cal_src.append([ix, iy])
                cal_dst.append([nx * SCREEN_W, ny * SCREEN_H])
                break
            if key == 27:  # Escape quits
                sys.exit("Calibration aborted")
    cv2.destroyWindow("Calibration")
    return np.array(cal_src), np.array(cal_dst)


# ------------------
# 3. Compute mapping
# ------------------


def solve_mapping(src, dst):
    # Want a mapping: [ix, iy, 1] @ M = [sx, sy]
    A = np.hstack([src, np.ones((src.shape[0], 1))])  # shape: N x 3
    # solve least squares for M: shape 3x2
    M, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)
    return M  # [ix, iy, 1] @ M = [screen_x, screen_y]


# ------------------------------------------
# 4. Drive real-time mouse-movement via gaze
# ------------------------------------------
def drive_mouse_via_gaze(cap, M):
    last_x, last_y = SCREEN_W / 2, SCREEN_H / 2
    alpha = 0.3  # smoothing factor

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            lx, ly = get_iris_center(lm, LEFT_IRIS_IDX, w, h)
            rx, ry = get_iris_center(lm, RIGHT_IRIS_IDX, w, h)
            ix, iy = (lx + rx) / 2, (ly + ry) / 2
            # map to screen (affine)
            src_v = np.array([ix, iy, 1.0])
            sx, sy = src_v @ M
            # smooth
            cx = last_x + alpha * (sx - last_x)
            cy = last_y + alpha * (sy - last_y)
            mouse_position = (cx, cy)
            last_x, last_y = cx, cy

        # OPTIONAL: display camera feed
        cv2.imshow("Mouse->Gaze", frame)
        key = cv2.waitKey(1)
        if key == 27:  # Esc to quit
            break


# -------
# 4. Main
# -------


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot open camera")
    print("Starting calibration...")
    src, dst = calibrate(cap)
    print("Calibrating mapping...")
    M = solve_mapping(src, dst)
    print("Starting gaze-controlled mouse. Press Esc to exit.")
    drive_mouse_via_gaze(cap, M)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
