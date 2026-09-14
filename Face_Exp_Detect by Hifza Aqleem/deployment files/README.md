# Expression Detection Hugging Face Space
This dynamic project focuses on training an AI-Based Facial Expression Detection system by utilizing the advanced computer vision framework Ultralytics YOLOv8 coupled with a custom computer vision dataset hosted on Roboflow.

Workflow Details. The structured workflow implemented within the laboratory notebook covers a complete machine learning lifecycle, split into the following operational stages:
Plain text Roboflow Dataset → Download YOLOv8 Format → Train YOLO → Evaluate → Test → Save Model.

Dataset Acquisition: Downloads version 4 of the facial-expression-rwslf dataset from Roboflow in the native YOLOv8 annotation format.

Data Profile & Composition: The structured image repository consists of a standard evaluation split detailing the following 
sample sizes: 
Train Images: 7,371, 
samples Validation Images: 692,
samples Test Images: 353
samples Target Classes (nc = 7): 
The neural network is designed to simultaneously look for and categorize 7 core human expressions:
[Angry, Disgust, Fear, Happiness, Neutral, Sadness, Surprised]. 

Technical Stack Used Base Framework: 
Ultralytics (YOLO).
Dataset Hosting & API: roboflow. 
Infrastructure: Google Colab Environment.
Supporting Libraries: os, glob, yaml, pandas, matplotlib, I Python.