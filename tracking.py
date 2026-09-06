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
