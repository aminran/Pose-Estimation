import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile

# -------------------------
# MediaPipe setup
# -------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b):
    radians = np.arctan2(a.y - b.y, a.x - b.x)
    angle = np.abs(radians * 180 / np.pi)
    return round(angle, 2)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🧘 Pose Rep Counter")

mode = st.radio("Select input source:", ["Upload Video", "Live Camera"])

# -------------------------
# Upload Video
# -------------------------
if mode == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
    if uploaded_file is not None:
        tfile = NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        cap = cv2.VideoCapture(video_path)

# -------------------------
# Live Camera
# -------------------------
if mode == "Live Camera":
    cap = cv2.VideoCapture(0)  # 0 = default camera

# -------------------------
# Video display placeholder
# -------------------------
frame_placeholder = st.empty()

# -------------------------
# Pose Estimation loop
# -------------------------
counter = 0
stage = None
if mode != None :
    with mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                angle = calculate_angle(hip, shoulder)

                h, w, _ = image.shape
                cv2.putText(
                    image,
                    str(angle),
                    tuple(np.multiply((hip.x, hip.y), [w, h]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # Counting logic
                if angle < 7:
                    stage = "down"
                if angle > 40 and stage == "down":
                    stage = "up"
                    counter += 1

            except:
                pass

            # UI overlay
            h, w, _ = image.shape
            cv2.rectangle(image, (0, 0), (200, 80), (255, 100, 0), -1)
            cv2.putText(image, "REPS:", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(image, str(counter), (100, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.rectangle(image, (w - 200, 0), (w, 80), (255, 100, 0), -1)
            cv2.putText(image, "Stage:", (w - 190, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(image, str(stage), (w - 90, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Convert BGR to RGB for Streamlit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(image_rgb, channels="RGB")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
