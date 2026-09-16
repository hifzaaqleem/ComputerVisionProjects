# ✊ ✋ ✌️ AI-Powered Rock, Paper, Scissors Game

A real-time **Rock, Paper, Scissors** game built using Python and Computer Vision. This project allows users to play the classic hand-gesture game directly against the computer using their webcam.

---

## 🚀 Features
- **Real-Time Detection:** Tracks hand gestures instantly using webcam feed.
- **Computer Vision Processing:** Utilizes advanced image processing/deep learning to accurately classify Rock, Paper, or Scissors.
- **Interactive Gameplay:** Play rounds continuously with live score tracking.
- **Lightweight & Fast:** Designed to run smoothly locally on standard hardware.

---

## 🛠️ Tech Stack
- **Programming Language:** Python 
- **Libraries & Frameworks:** 
  - `OpenCV` (for video capture and image manipulation)
  - `MediaPipe` / `TensorFlow` / `PyTorch` *(Update based on whether you used MediaPipe landmarks or a custom CNN model)*
  - `NumPy`

---

## 📁 Project Structure
```text
ComputerVisionProjects/
│
└── rock_paper_scissors/
    ├── app.py              # Main application script
    ├── model/              # (Optional) Trained model weights or cascades
    └── requirements.txt    # Project dependencies
