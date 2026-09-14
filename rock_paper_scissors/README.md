# RPS Vision Arena — Version 2

This app uses the trained YOLO `best.pt` from the supplied
Rock Paper Scissors notebook.

## Your notebook model

The notebook reports:
- Classes: Paper, Rock, Scissors
- Epochs: 15
- Image size: 640
- Precision: 0.943
- Recall: 0.891
- mAP50: 0.937
- mAP50-95: 0.731

The trained weights are produced at:

`/content/runs/detect/rock-paper-scissors-roboflow/weights/best.pt`

Download `best.pt` from Colab and put it beside `app.py`.

## Install

```bash
pip install -r requirements.txt
```

For a local Streamlit run:

```bash
streamlit run app.py
```

For the OpenCV live demo:

```bash
python live_video_demo.py
```

Press Q to stop the OpenCV demo.

## Version 2 features

- Animated robot opponent
- Browser webcam
- YOLO bounding boxes
- Rock / Paper / Scissors prediction
- Confidence display
- Automatic AI move
- Winner calculation
- Score tracking
- Round counter
- Countdown/progress bar
- Image testing tab
- Model information tab

## Important deployment note

For hosted Streamlit, browser webcam access normally requires HTTPS.
Localhost works with normal browser camera permissions.

## Suggested project structure

```text
RPS_Vision_Arena/
├── app.py
├── live_video_demo.py
├── requirements.txt
└── best.pt
```
