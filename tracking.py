import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

### the video file to process
video_path = "videos/live_out.mov"     
path_out = "best_match_zoom.jpg"

### Define the target attributes for the person we want to find
GENDER = "a photo of a man"
COLOR = "wearing light blue clothing"
CLOTHES = "wearing a shirt"

### Define the choices for each attribute
GENDER_choice = [GENDER, "a photo of a man"]
COLOR_choice = [COLOR, "wearing black clothing", "wearing white clothing", "wearing red clothing"]
CLOTHES_choice = [CLOTHES, "wearing pants", "wearing a dress", "wearing a jacket"]

### creating the output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

### models (yolo for detection and CLIP for classification)
yolo_model = YOLO("yolov8n.pt") 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
