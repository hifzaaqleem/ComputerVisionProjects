import gradio as gr
from PIL import Image
from ultralytics import YOLO

# Load the fine-tuned YOLO model
MODEL_PATH = "best.pt"  # Update path to 'weights/best.pt' if using subfolder
model = YOLO(MODEL_PATH)


def predict_objects(image: Image.Image, conf_threshold: float = 0.4):
    """Runs YOLO object detection on the input image and returns the annotated image."""
    if image is None:
        return None

    # Perform inference
    results = model.predict(source=image, conf=conf_threshold)[0]

    # Convert BGR (OpenCV format output by YOLO plot) to RGB for PIL/Gradio
    annotated_array = results.plot()[..., ::-1]
    return Image.fromarray(annotated_array)


# Build Gradio Web Interface
iface = gr.Interface(
    fn=predict_objects,
    inputs=[
        gr.Image(type="pil", label="Upload Store Image"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.4,
            step=0.05,
            label="Confidence Threshold",
        ),
    ],
    outputs=gr.Image(type="pil", label="Detected Objects"),
    title="🛒 In-Store Human Detection",
    description=(
        "An AI-powered computer vision application designed for retail environments. "
        "Upload a retail or store image to detect people and objects in real time using a fine-tuned YOLOv8 model."
    ),
    examples=[],
    theme="soft",
)

if __name__ == "__main__":
    iface.launch()