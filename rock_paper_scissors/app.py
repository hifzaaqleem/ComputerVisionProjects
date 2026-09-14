import os
import random
import time
import cv2
import av
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(
    page_title="RPS Vision Arena",
    page_icon="✊",
    layout="wide",
)

# ============================================================
# PAGE STYLE
# ============================================================
st.markdown("""
<style>
.hero {
    padding: 22px 28px;
    border-radius: 22px;
    border: 1px solid rgba(128,128,128,.25);
    background: linear-gradient(135deg, rgba(90,90,160,.16), rgba(50,170,180,.10));
    margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 42px; }
.hero p { margin: 5px 0 0; opacity: .75; font-size: 17px; }

.card {
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.25);
    background: rgba(128,128,128,.055);
}

.robot {
    font-size: 105px;
    text-align: center;
    animation: float 1.6s ease-in-out infinite alternate;
}
@keyframes float {
    from { transform: translateY(0px) rotate(-2deg); }
    to   { transform: translateY(-13px) rotate(2deg); }
}

.move {
    font-size: 44px;
    text-align: center;
    padding: 10px;
}
.result {
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    padding: 14px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,.25);
}
.small { opacity: .72; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>✊ RPS Vision Arena</h1>
<p>Real-time YOLO Rock • Paper • Scissors detection + interactive AI opponent</p>
</div>
""", unsafe_allow_html=True)

GESTURES = ["Rock", "Paper", "Scissors"]
EMOJI = {"Rock": "✊", "Paper": "✋", "Scissors": "✌️"}

# Your notebook uses:
# Paper, Rock, Scissors
DEFAULT_MODEL = "best.pt"


def outcome(player, ai):
    if player == ai:
        return "DRAW"
    if (
        (player == "Rock" and ai == "Scissors")
        or (player == "Paper" and ai == "Rock")
        or (player == "Scissors" and ai == "Paper")
    ):
        return "YOU WIN"
    return "AI WINS"


@st.cache_resource
def load_model(path):
    return YOLO(path)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Model Settings")

    uploaded = st.file_uploader(
        "Upload trained YOLO model",
        type=["pt"],
        help="Use the best.pt produced by your notebook."
    )

    if uploaded:
        model_path = "/tmp/rps_best.pt"
        with open(model_path, "wb") as f:
            f.write(uploaded.getbuffer())
    else:
        model_path = DEFAULT_MODEL

    confidence = st.slider(
        "Confidence threshold",
        0.10, 0.95, 0.40, 0.05
    )

    st.divider()
    st.write("**Model classes**")
    st.write("✊ Rock")
    st.write("✋ Paper")
    st.write("✌️ Scissors")

    st.divider()
    st.caption(
        "The supplied notebook trains a YOLO detector for Paper, Rock "
        "and Scissors at 640px for 15 epochs."
    )

if not os.path.exists(model_path):
    st.warning("Put `best.pt` beside `app.py`, or upload it in the sidebar.")
    st.info("Notebook output path: /content/runs/detect/rock-paper-scissors-roboflow/weights/best.pt")
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ============================================================
# VIDEO PROCESSOR
# ============================================================
class RPSProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.conf = 0.40

        # Game state
        self.game_mode = False
        self.ai_move = random.choice(GESTURES)
        self.player_move = "Waiting..."
        self.last_move = None
        self.round_start = time.time()
        self.result = "Make your move!"
        self.you_score = 0
        self.ai_score = 0
        self.draws = 0
        self.round_number = 1
        self.last_scored_round = 0

        # Detection state
        self.label = "No detection"
        self.score = 0.0

    def set_model(self, model):
        self.model = model

    def set_conf(self, conf):
        self.conf = conf

    def start_game(self):
        self.game_mode = True
        self.you_score = 0
        self.ai_score = 0
        self.draws = 0
        self.round_number = 1
        self.last_scored_round = 0
        self.new_round()

    def stop_game(self):
        self.game_mode = False

    def new_round(self):
        self.ai_move = random.choice(GESTURES)
        self.player_move = "Waiting..."
        self.last_move = None
        self.result = "Show your hand!"
        self.round_start = time.time()

    def detect(self, frame):
        if self.model is None:
            return frame, "No detection", 0.0

        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=640,
            verbose=False
        )[0]

        annotated = result.plot()

        best_name = "No detection"
        best_conf = 0.0

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                c = float(box.conf[0])
                cls_id = int(box.cls[0])
                name = result.names.get(cls_id, str(cls_id))

                if c > best_conf:
                    best_conf = c
                    best_name = name

        return annotated, best_name, best_conf

    def draw_text(self, frame, text, xy, scale=0.7, thickness=2):
        cv2.putText(
            frame, text, xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale, (255, 255, 255),
            thickness, cv2.LINE_AA
        )

    def game_overlay(self, frame, detected):
        h, w = frame.shape[:2]

        # Start a new round after the previous result.
        elapsed = time.time() - self.round_start

        if elapsed >= 5.0:
            self.round_number += 1
            self.new_round()
            elapsed = 0.0

        # Score only once, after one second of stable gameplay.
        if (
            detected in GESTURES
            and self.last_scored_round != self.round_number
            and elapsed >= 1.0
        ):
            self.player_move = detected
            self.last_move = detected
            self.result = outcome(detected, self.ai_move)
            self.last_scored_round = self.round_number

            if self.result == "YOU WIN":
                self.you_score += 1
            elif self.result == "AI WINS":
                self.ai_score += 1
            else:
                self.draws += 1

        # Top game banner
        cv2.rectangle(frame, (0, 0), (w, 112), (22, 22, 22), -1)

        self.draw_text(
            frame,
            f"ROUND {self.round_number}",
            (20, 35),
            0.75,
            2
        )

        self.draw_text(
            frame,
            f"YOU {self.you_score}   -   {self.ai_score} AI   |   DRAWS {self.draws}",
            (20, 78),
            0.72,
            2
        )

        # Bottom panel
        y = max(115, h - 145)
        cv2.rectangle(frame, (0, y), (w, h), (25, 25, 25), -1)

        player_display = self.player_move
        ai_display = self.ai_move

        self.draw_text(
            frame,
            f"YOU: {player_display}",
            (20, y + 42),
            0.80,
            2
        )
        self.draw_text(
            frame,
            f"AI:  {ai_display}",
            (20, y + 84),
            0.80,
            2
        )

        if self.last_move:
            self.draw_text(
                frame,
                self.result,
                (w - 245, y + 62),
                0.82,
                2
            )

        # Confidence
        self.draw_text(
            frame,
            f"YOLO: {detected} ({self.score * 100:.1f}%)",
            (w - 410, 35),
            0.60,
            1
        )

        # Countdown/progress
        remaining = max(0.0, 5.0 - elapsed)
        bar = int((remaining / 5.0) * max(1, w - 40))
        cv2.rectangle(
            frame, (20, h - 12), (w - 20, h - 5), (80, 80, 80), -1
        )
        cv2.rectangle(
            frame, (20, h - 12), (20 + bar, h - 5), (255, 255, 255), -1
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        annotated, label, score = self.detect(image)

        self.label = label
        self.score = score

        if self.game_mode:
            self.game_overlay(annotated, label)
        else:
            # Detection-only overlay
            cv2.rectangle(
                annotated, (10, 10), (470, 76), (20, 20, 20), -1
            )
            self.draw_text(
                annotated,
                f"Prediction: {label}",
                (25, 42),
                0.82,
                2
            )
            self.draw_text(
                annotated,
                f"Confidence: {score * 100:.1f}%",
                (25, 66),
                0.52,
                1
            )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ============================================================
# TABS
# ============================================================
tab_game, tab_live, tab_image, tab_about = st.tabs(
    ["🎮 AI Game", "🎥 Live Detection", "🖼️ Image Test", "📊 About Model"]
)

# ============================================================
# GAME
# ============================================================
with tab_game:
    st.subheader("🎮 Play against the animated AI opponent")

    left, right = st.columns([1, 2])

    with left:
        st.markdown(
            '<div class="robot">🤖</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="small" style="text-align:center;">NOVA RPS BOT</div>',
            unsafe_allow_html=True
        )

        if "game_started" not in st.session_state:
            st.session_state.game_started = False

        start = st.button(
            "▶️ Start Game",
            use_container_width=True,
            type="primary"
        )

        if start:
            st.session_state.game_started = True
            st.rerun()

        st.markdown("---")
        st.markdown("### How to play")
        st.write("1. Start the game.")
        st.write("2. Allow camera access.")
        st.write("3. Show Rock, Paper or Scissors.")
        st.write("4. YOLO detects your movement.")
        st.write("5. The AI selects a move.")
        st.write("6. The winner and score appear.")

    with right:
        if not st.session_state.game_started:
            st.info("Click **Start Game** to begin.")
        else:
            ctx_game = webrtc_streamer(
                key="rps-game",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=RPSProcessor,
                media_stream_constraints={
                    "video": True,
                    "audio": False
                },
                async_processing=True,
            )

            if ctx_game.video_processor:
                ctx_game.video_processor.set_model(model)
                ctx_game.video_processor.set_conf(confidence)

                if not hasattr(ctx_game.video_processor, "_game_initialized"):
                    ctx_game.video_processor.start_game()
                    ctx_game.video_processor._game_initialized = True

                p = ctx_game.video_processor

                c1, c2, c3 = st.columns(3)
                c1.metric("You", p.you_score)
                c2.metric("AI", p.ai_score)
                c3.metric("Draws", p.draws)

                if p.last_move:
                    st.markdown(
                        f'<div class="result">{EMOJI[p.last_move]} {p.last_move} '
                        f'vs {EMOJI[p.ai_move]} {p.ai_move} → {p.result}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Show your hand to make a move.")

# ============================================================
# LIVE DETECTION
# ============================================================
with tab_live:
    st.subheader("🎥 Live YOLO detection")

    ctx = webrtc_streamer(
        key="rps-live-only",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=RPSProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.set_model(model)
        ctx.video_processor.set_conf(confidence)

        if ctx.video_processor.label in GESTURES:
            st.success(
                f"{EMOJI[ctx.video_processor.label]} "
                f"{ctx.video_processor.label} — "
                f"{ctx.video_processor.score * 100:.1f}%"
            )
        else:
            st.info("Show your hand to the camera.")

# ============================================================
# IMAGE
# ============================================================
with tab_image:
    st.subheader("🖼️ Test the trained detector on an image")

    image_file = st.file_uploader(
        "Upload JPG / PNG / WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        key="image-test"
    )

    if image_file:
        data = np.frombuffer(image_file.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        result = model.predict(
            image,
            conf=confidence,
            imgsz=640,
            verbose=False
        )[0]

        annotated = result.plot()

        a, b = st.columns([2, 1])

        with a:
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        with b:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = result.names.get(cls_id, str(cls_id))
                    st.metric(name, f"{conf * 100:.1f}%")
            else:
                st.warning("No gesture detected.")

# ============================================================
# ABOUT
# ============================================================
with tab_about:
    st.subheader("📊 Your trained model")

    st.markdown("""
    <div class="card">
    <h3>Rock • Paper • Scissors YOLO detector</h3>
    <p>
    The application uses the trained <b>best.pt</b> detector from the supplied
    notebook. The notebook trains three object-detection classes:
    <b>Paper, Rock, Scissors</b>.
    </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Classes", "3")
    c2.metric("Training epochs", "15")
    c3.metric("Image size", "640")

    st.markdown("### Validation reported by the notebook")
    st.write("- Precision: 0.943")
    st.write("- Recall: 0.891")
    st.write("- mAP50: 0.937")
    st.write("- mAP50-95: 0.731")

    st.caption(
        "These values are taken from the validation output in the supplied notebook."
    )
