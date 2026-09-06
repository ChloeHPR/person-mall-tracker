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


### video analysis and treatment 
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) ### give the total number of frame for the analysis
#frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # end of the video in this case
        
    #frame_count += 1

    # Detect person using class 0 with YOLO 
    results = yolo_model(frame, classes=[0], conf=0.5, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) > 0:
        ### convert BGR to RGB for CLIP
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            ### bounding box conditions
            if x2 > x1 and y2 > y1:
                ### to get only one person in a screenshot
                crop = pil_image.crop((x1, y1, x2, y2))
                
                ### evaluate attributes
                inputs_gender = clip_processor(text=GENDER_choixe, images=crop, return_tensors="pt", padding=True).to(device)
                
                
### save score and most matched person
              
    
cap.release()
