# person-mall-tracker

This computer vision project allows you to find a specific person in a video stream using a description of their gender and clothing attributes. 

Designed to be versatile, it uses a hybrid approach combining object detection and semantic analysis.

## Technical Architecture

To avoid the biases of purely generalist models, this script implements **Person Attribute Recognition (PAR)**:
1. **YOLOv8**: Detects and extracts (crops) all human silhouettes present in each frame.
2. **OpenAI CLIP**: Evaluates each silhouette independently across 3 strict categories (Gender, Clothing Color, Clothing Type) using a Zero-Shot Classification approach.
3. **Scoring**: The probabilities are averaged. The script stores in memory the cropped image that achieved the highest score across the entire video.

## Installation

1. Clone this repository
2. Install the dependencies
   ```bash
   pip install -r requirements.txt
   ```


## Main usage

 * Place the video you want to analyze in the videos/ folder and name it live_out.mp4 (or update the file path in tracker.py using your own path).
* Modify the target variables in tracking.py to configure your search parameters for example like this:
 ```bash
GENDER = "a photo of a woman"
COLOR= "wearing red clothing"
CLOTHE = "wearing a dress"
  ```
* run the script :
  ```bash
  python tracking.py
  ```
  (Sometimes you need to specify the version of python you use, in my case I put python3)

The final result (a cropped screenshot of the best-matching person) will be automatically saved as best_match_crop.jpg in your main folder. 

### About performance

The script automatically detects your available hardware (CPU, NVIDIA CUDA, or Apple Silicon MPS) to drastically accelerate computations during video processing.
