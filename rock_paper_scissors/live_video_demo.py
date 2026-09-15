import cv2
from ultralytics import YOLO

model_path = "best (1).pt"
CONFIDENCE = 0.40

model = YOLO(model_path)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Webcam could not be opened. Check camera permissions."
    )

print("RPS live detection started. Press Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model.predict(
        frame,
        conf=CONFIDENCE,
        imgsz=640,
        verbose=False
    )[0]

    annotated = result.plot()

    label = "No detection"
    score = 0.0

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = result.names.get(cls_id, str(cls_id))

            if conf > score:
                score = conf
                label = name

    cv2.rectangle(annotated, (10, 10), (480, 80), (20, 20, 20), -1)

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
        (25, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    cv2.imshow("RPS - Live YOLO Demo", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
