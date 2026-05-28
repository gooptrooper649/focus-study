import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO
import pygame

# ------------------ INIT ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

model = YOLO("yolov8n.pt")  # lightweight model

pygame.mixer.init()

pygame.mixer.init()
beep_sound = pygame.mixer.Sound("beep.wav")

def beep():
    beep_sound.play()
    
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

def get_eye_center(landmarks, eye_indices, w, h):
    x = int((landmarks[eye_indices[0]].x + landmarks[eye_indices[1]].x) / 2 * w)
    y = int((landmarks[eye_indices[0]].y + landmarks[eye_indices[1]].y) / 2 * h)
    return x, y

# ------------------ STATE ------------------
calibrated = False
ref_point = None

distraction_start = None
DISTRACTION_THRESHOLD = 3
GAZE_TOLERANCE = 60

# Pomodoro
WORK_TIME = 45 * 60
BREAK_TIME = 15 * 60
session_start = time.time()
on_break = False

# Grace allowance
GRACE_LIMIT = 60  # total seconds
grace_used = 0

# ------------------ LOOP ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ------------------ POMODORO ------------------
    elapsed_session = time.time() - session_start

    if not on_break and elapsed_session > WORK_TIME:
        on_break = True
        session_start = time.time()
    elif on_break and elapsed_session > BREAK_TIME:
        on_break = False
        session_start = time.time()

    # ------------------ FACE TRACK ------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    distracted = False

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_eye = get_eye_center(landmarks, LEFT_EYE, w, h)
            right_eye = get_eye_center(landmarks, RIGHT_EYE, w, h)

            center_x = (left_eye[0] + right_eye[0]) // 2
            center_y = (left_eye[1] + right_eye[1]) // 2

            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

            if calibrated and not on_break:
                dist = np.linalg.norm(np.array([center_x, center_y]) - np.array(ref_point))

                if dist > GAZE_TOLERANCE:
                    distracted = True

    # ------------------ PHONE DETECTION ------------------
    phone_detected = False
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "cell phone":
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, "PHONE!", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # ------------------ DECISION ENGINE ------------------
    status_text = "FOCUSED"

    if on_break:
        status_text = "BREAK TIME"
    else:
        if phone_detected:
            status_text = "PHONE DETECTED!"
            beep()
        elif distracted:
            if grace_used < GRACE_LIMIT:
                grace_used += 1
                status_text = f"GRACE {grace_used}s"
            else:
                status_text = "DISTRACTED!"
                beep()

    # ------------------ UI ------------------
    cv2.putText(frame, status_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0) if "FOCUSED" in status_text else (0,0,255), 2)

    # Timer display
    remaining = int((WORK_TIME if not on_break else BREAK_TIME) - elapsed_session)
    cv2.putText(frame, f"Time: {remaining}s", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Calibration prompt
    if not calibrated:
        cv2.putText(frame, "Press C to calibrate", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.imshow("Focus AI", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and result.multi_face_landmarks:
        ref_point = (center_x, center_y)
        calibrated = True
        grace_used = 0
        print("Calibrated")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()