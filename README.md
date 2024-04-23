# Pedestrian Detection Project

This project demonstrates real-time pedestrian detection using the YOLOv4-tiny object detection model with OpenCV. It processes video frames, detects pedestrians, and annotates them with bounding boxes.

## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- imutils (`pip install imutils`)

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/pedestrian-detection.git
   cd pedestrian-detection
   ```
### 2.Download YOLOv4-tiny model files:
- Download yolov4-tiny.weights and yolov4-tiny.cfg from the official YOLO website or other sources.
- Save these files into the project directory.

### 3.Obtain the COCO labels file (coco.names):
- The coco.names file contains a list of class names recognized by the YOLO model, including "person".
- Place coco.names in the project directory.

## Usage

1.Run the pedestrian detection script:
```bash
python pedestrian_detection.py
```
2. Script Behaviour
- The script opens a video file and starts processing frames for pedestrian detection.
- Detected pedestrians are annotated with bounding boxes.
- Press the Esc key to exit the application.

## Customization

- Input Video: Modify the video_path variable in the script (pedestrian_detection.py) to use a different video file.
- Detection Parameters: Adjust NMS_THRESHOLD and MIN_CONFIDENCE constants in the script for controlling detection sensitivity.
- Model Configuration: Customize the YOLO model by changing the configPath and weightsPath variables in the script.
## Acknowledgments

- This project uses YOLOv4-tiny, a deep learning model developed by the Darknet team. Visit the official YOLO website for more information.
- The COCO dataset and labels (coco.names) are used for object recognition. Refer to the COCO website for dataset details.

   
