import os
import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="RPS Live YOLO",
    page_icon="✊",
    layout="wide"
)

st.title("✊ ✋ ✌️ RPS Live YOLO WebRTC Demo")
st.write(
    "Allow camera access below to stream live Rock, Paper, "
    "Scissors detection."
)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

CONFIDENCE = 0.40


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return YOLO(MODEL_PATH)


model = load_model()


# ------------------------------------------------------------
# YOLO VIDEO PROCESSOR
# ------------------------------------------------------------
class YOLOVideoTransformer:

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        # Convert WebRTC frame to OpenCV BGR image
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO
        result = model.predict(
            img,
            conf=CONFIDENCE,
            imgsz=640,
            verbose=False
        )[0]

        # Draw YOLO bounding boxes
        annotated = result.plot()

        # Default values
        label = "No detection"
        score = 0.0

        # Find strongest detection
        if result.boxes is not None and len(result.boxes) > 0:

            for box in result.boxes:

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                name = result.names.get(
                    cls_id,
                    str(cls_id)
                )

                if conf > score:
                    score = conf
                    label = name

        # ----------------------------------------------------
        # INFO PANEL
        # ----------------------------------------------------
        cv2.rectangle(
            annotated,
            (10, 10),
            (480, 85),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            annotated,
            f"Prediction: {label}",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated,
            f"Confidence: {score * 100:.1f}%",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# ------------------------------------------------------------
# WEBRTC CAMERA
# ------------------------------------------------------------
webrtc_streamer(
    key="rps-live",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOVideoTransformer,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

st.sidebar.write(
    f"Confidence threshold: {CONFIDENCE:.2f}"
)

st.sidebar.write(
    "Classes: Paper • Rock • Scissors"
)

st.sidebar.success("YOLO model loaded successfully.")
```
