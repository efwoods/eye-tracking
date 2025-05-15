# eye-tracking

This is a proof-of-concept of eye tracking software to cotrol a mouse with vision

This software will use a webcame to track gaze and determine where on the screen the individual is looking and move the mouse accordingly in real-time

---
Below is a fully self-contained, calibration-free proof-of-concept in Python. It uses only open-source, local libraries (OpenCV, MediaPipe, NumPy) and no per-user calibration steps. The pipeline is:

    FaceMesh → 2D landmarks (Mediapipe)

    SolvePnP on a generic 3D face model → head pose (rotation R, translation T) cite[1]

    Unproject iris center from image to a 3D gaze ray in camera coordinates

    Rotate that ray into head coordinates (by Rᵀ) → approximate gaze in head space

    Intersect that ray with a virtual plane at your screen distance → (X,Y) on plane → map to your screen pixels


```python

import cv2
import numpy as np
import mediapipe as mp
from pynput.mouse import Controller

# 1. Setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)
mouse = Controller()

# approximate screen distance in millimeters
SCREEN_DIST = 546.1  
# physical screen size in mm (width, height) and resolution
PHYS_W, PHYS_H = 531.0, 298.0   # e.g. 24" 16:9
RES_X, RES_Y = 3440, 1440

# Mediapipe landmark indices
LEFT_EYE = [33, 133]   # left eye corners
RIGHT_EYE = [362, 263] # right eye corners
LEFT_IRIS = [473, 474, 475, 476]
RIGHT_IRIS = [468, 469, 470, 471]
NOSE_TIP = 1

# generic 3D model points in mm (X,Y,Z) for SolvePnP
MODEL_3D = np.array([
    [-32.0,   40.0,   30.0],   # left eye outer corner
    [ 32.0,   40.0,   30.0],   # right eye outer corner
    [  0.0,    0.0,    0.0],   # nose tip
], dtype=np.float32)
# corresponding 2D landmark indices
MODEL_2D_IDX = [33, 263, 1]

def get_avg(landmarks, idxs, w, h):
    xs = [landmarks[i].x * w for i in idxs]
    ys = [landmarks[i].y * h for i in idxs]
    return np.mean(xs), np.mean(ys)

# build camera intrinsics from frame size
def get_camera_matrix(w, h):
    # focal length ~ w in pixels
    f = w
    return np.array([[f, 0, w/2],
                     [0, f, h/2],
                     [0, 0,   1]], dtype=np.float32)

def intersect_plane(ray_o, ray_d, plane_z):
    # solve o + t d → z = plane_z → t = (plane_z - oz) / dz
    t = (plane_z - ray_o[2]) / ray_d[2]
    return ray_o + t * ray_d

# 2. Main loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    cam_mat = get_camera_matrix(w, h)
    dist_coeffs = np.zeros((4,1))  # assume no lens distortion

    # detect landmarks
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        cv2.imshow("Gaze", frame)
        if cv2.waitKey(1)==27: break
        continue

    lm = res.multi_face_landmarks[0].landmark

    # 2D points for SolvePnP
    pts2d = np.array([[lm[i].x*w, lm[i].y*h] for i in MODEL_2D_IDX], dtype=np.float32)

    # SolvePnP → head rotation (R) & translation (T)
    _, rvec, tvec = cv2.solvePnP(
        MODEL_3D, pts2d, cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    R, _ = cv2.Rodrigues(rvec)    # 3×3 rotation matrix
    T = tvec.reshape(3)           # translation vector

    # get iris image position (average of both eyes)
    lx, ly = get_avg(lm, LEFT_IRIS, w, h)
    rx, ry = get_avg(lm, RIGHT_IRIS, w, h)
    ix, iy = (lx+rx)/2, (ly+ry)/2

    # unproject to camera ray: K^{-1} [ix,iy,1]^T
    uv1 = np.array([ix, iy, 1.0], dtype=np.float32)
    cam_ray = np.linalg.inv(cam_mat) @ uv1
    cam_ray /= np.linalg.norm(cam_ray)

    # convert to head coordinates: R^T (cam_ray)
    head_ray = R.T @ cam_ray

    # ray origin in head coords is (0,0,0) at nose tip; translate to camera coords
    cam_origin = (R @ np.array([0,0,0],float)) + T
    # intersect with plane at Z = SCREEN_DIST in camera coords
    pt3d = intersect_plane(cam_origin, cam_ray, SCREEN_DIST)

    # map from physical mm on plane to pixel coordinates
    x_mm = pt3d[0];  y_mm = pt3d[1]
    px = (x_mm + PHYS_W/2)  * (RES_X/PHYS_W)
    py = (PHYS_H/2 - y_mm)  * (RES_Y/PHYS_H)

    # move the mouse
    mouse.position = (px, py)

    # optional: visualize
    cv2.circle(frame, (int(ix),int(iy)), 5, (0,255,0), -1)
    cv2.imshow("Gaze", frame)
    if cv2.waitKey(1) == 27:  # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()

```
