import os
# Force Ultralytics to use a writable directory immediately
os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'

import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np # Needed for video processing

# -------------------------------------------------------------------------
# 1. Load your model (Keep your existing initialization)
# -------------------------------------------------------------------------
# Make sure 'best (1).pt' is uploaded in your Space!
model = YOLO("best (1).pt")

# Explicitly map the class names from your custom notebook training run
class_names = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprised']


# -------------------------------------------------------------------------
# 2. YOUR ORIGINAL PREDICTION LOGIC (Kept exactly as provided)
# -------------------------------------------------------------------------
def predict_image(image):
    if image is None:
        return None, "Please upload an image."
        
    # Convert image from RGB (Gradio default) to BGR (YOLO default)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    # Run prediction with a lowered confidence threshold to catch expressions easily
    results = model(img_bgr, conf=0.15)
        
    # Plot the bounding boxes onto the image
    annotated = results[0].plot()
        
    # Convert back to RGB for Gradio output presentation
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
    # Extract the names of the expressions found using your model classes
    labels = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        # Use explicit mapping if internal dictionary is rebuilding
        if class_id < len(class_names):
            labels.append(class_names[class_id])
        else:
            labels.append(model.names[class_id])
            
    detected_text = ", ".join(labels) if labels else "No expressions detected"
    
    # Return both the image (numpy) and the labels string
    return annotated_rgb, detected_text


# -------------------------------------------------------------------------
# 3. NEW: VIDEO PROCESSING LOGIC (Processes frame-by-frame)
# -------------------------------------------------------------------------
def process_video_file(video_path):
    """
    Reads an uploaded video file, processes it using the existing 
    YOLO prediction logic, and outputs a complete annotated video.
    """
    if video_path is None:
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None # Could not open video
        
    # Get properties to set up output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 # Fallback if FPS cannot be read
    
    output_path = "processed_expression_video.mp4"
    
    # Set up the VideoWriter. 'mp4v' is generally safe for web display.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Start processing video: {video_path}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Convert frame to RGB for processing logic
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Reuse your image processing logic (we only need the image output)
        processed_rgb, _ = predict_image(frame_rgb)
        
        if processed_rgb is not None:
            # 3. Convert back to BGR for video writer
            frame_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        else:
            # If prediction fails, just write original frame (rare)
            out.write(frame)
            
    cap.release()
    out.release()
    print("Video processing complete.")
    return output_path


# -------------------------------------------------------------------------
# 4. UPDATED: GRADIO APP LAYOUT (Unified Image, Video, Webcam)
# -------------------------------------------------------------------------

# Create a custom Blocks layout
with gr.Blocks(title="Face Expression Detection Suite", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🎭 Face Expression Detection")
    gr.Markdown(
        "Detect facial expressions (Angry, Disgust, Fear, Happiness, Neutral, Sadness, Surprised) "
        "using your custom trained YOLOv8 model."
    )
    gr.Markdown("Select an input source below:")

    with gr.Tabs():
        
        # --- TAB 1: STATIC IMAGE (Your original interface functionality) ---
        with gr.TabItem("🖼️ Image Upload"):
            with gr.Row():
                img_input = gr.Image(type="numpy", label="Upload Face Image")
                img_output_img = gr.Image(type="numpy", label="Detection Results")
            img_output_text = gr.Textbox(label="Detected Expressions")
            img_button = gr.Button("Analyze Image")
            
            # Action: Reuses your core logic
            img_button.click(
                fn=predict_image, 
                inputs=img_input, 
                outputs=[img_output_img, img_output_text]
            )
            
        # --- TAB 2: LIVE WEBCAM (Continuous prediction) ---
        with gr.TabItem("🎥 Live Webcam"):
            gr.Markdown("Show your face to the camera for real-time updates.")
            with gr.Row():
                # Setting 'streaming=True' creates a continuous stream
                webcam_input = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Live Feed")
                webcam_output_img = gr.Image(type="numpy", label="Real-time Detection")
            webcam_output_text = gr.Textbox(label="Current Expression")
            
            # Action: Triggers 'predict_image' continuously on frame change
            webcam_input.stream(
                fn=predict_image, 
                inputs=webcam_input, 
                outputs=[webcam_output_img, webcam_output_text]
            )

        # --- TAB 3: VIDEO FILE (Batch processing) ---
        with gr.TabItem("🎞️ Video File"):
            gr.Markdown("Upload a recorded video (.mp4 or .avi) to process all frames. This may take some time depending on video length.")
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
                video_output = gr.Video(label="Processed Video Output")
            video_button = gr.Button("Process Full Video")
            
            # Action: Uses the new frame-by-frame loop
            video_button.click(
                fn=process_video_file, 
                inputs=video_input, 
                outputs=video_output
            )

# -------------------------------------------------------------------------
# 5. Launch the combined app
# -------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()