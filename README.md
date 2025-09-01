# Face Mask Detection Project

A real-time face mask detection system that can analyze video feeds to determine whether detected faces are wearing masks or not. The project combines OpenCV's DNN face detection with custom mask classification logic.

## 🎯 Project Overview

This project provides two main components:
- **Real-time video analysis** (`video_mask.py`) - Processes video files or camera feeds to detect faces and classify mask-wearing status
- **ML model wrapper** (`mask_detector.py`) - A flexible class for integrating TensorFlow SavedModel-based mask detection

## 🛠️ Models Used

### 1. Face Detection Model
- **Model**: SSD MobileNet-based face detector
- **Files**: 
  - `res10_300x300_ssd_iter_140000.caffemodel` - Pre-trained weights
  - `deploy.prototxt` - Network architecture configuration
- **Purpose**: Detects and localizes faces in video frames
- **Input**: 300x300 RGB images
- **Output**: Bounding boxes with confidence scores for detected faces

### 2. Mask Classification
- **Current Implementation**: Standard deviation-based texture analysis
- **Logic**: Masked faces typically have lower texture variation in the lower face region
- **Threshold**: Configurable `COVERED_STD_THRESHOLD` (default: 30)
- **Alternative**: The `MaskDetector` class supports TensorFlow SavedModel integration for more sophisticated ML-based classification

## 📋 Prerequisites

- Python 3.7+
- Webcam or video file for testing

## 🚀 Installation & Setup

### 1. Clone/Download the Project
```bash
git clone <your-repo-url>
cd mask-detection-project
```

### 2. Install Dependencies
```bash
pip install opencv-python numpy tensorflow keras
```

### 3. Download Required Model Files
Download the face detection model files and place them in your project directory:

**Option A: Direct Download**
```bash
# Download face detection model
wget https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt
wget https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel
```

**Option B: Manual Download**
1. Download `deploy.prototxt` from [OpenCV GitHub](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
2. Download `res10_300x300_ssd_iter_140000.caffemodel` from [OpenCV GitHub](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel)

### 4. Prepare Test Video
- Place your test video file in the `lib/` directory as `sample2.mp4`
- Or modify the video path in `video_mask.py` to point to your video file

## 🎮 Usage

### Running Video Mask Detection
```bash
python video_mask.py
```

**Controls:**
- Press `q` to quit the application
- The detection window is resizable (800x600 default)

### Using the MaskDetector Class
```python
from mask_detector import MaskDetector
import numpy as np

# Initialize with your trained model path
detector = MaskDetector("path/to/your/saved_model")

# Prepare your image (preprocessed numpy array)
image = np.array(...)  # Shape: (height, width, channels) or (1, height, width, channels)

# Get prediction
result = detector.predict(image)
print(result)
```

## 📁 Project Structure
```
mask-detection-project/
├── mask_detector.py          # ML model wrapper class
├── video_mask.py            # Real-time video detection
├── deploy.prototxt          # Face detection model config
├── res10_300x300_ssd_iter_140000.caffemodel  # Face detection weights
├── lib/
│   └── sample2.mp4          # Test video file
└── README.md               # This file
```

## ⚙️ Configuration

### Video Detection Parameters
In `video_mask.py`, you can adjust:

```python
CONFIDENCE_THRESHOLD = 0.5      # Face detection confidence (0.0-1.0)
COVERED_STD_THRESHOLD = 30      # Mask detection sensitivity
DISPLAY_WIDTH = 800            # Output window width
DISPLAY_HEIGHT = 600           # Output window height
```

### Input Sources
- **Video file**: Modify the path in `cv2.VideoCapture("lib/sample2.mp4")`
- **Webcam**: Change to `cv2.VideoCapture(0)` for default camera

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
```bash
# If you get keras/tensorflow import errors:
pip install --upgrade tensorflow keras

# For OpenCV issues:
pip install opencv-python-headless  # For servers without GUI
```

**Model File Errors**
- Ensure both `.caffemodel` and `.prototxt` files are in the same directory as the script
- Check file permissions and paths

**Video File Issues**
- Verify video file exists at specified path
- Try with different video formats (MP4, AVI, MOV)
- For webcam: ensure camera permissions are granted

**Performance Issues**
- Reduce `DISPLAY_WIDTH` and `DISPLAY_HEIGHT` for better performance
- Lower `CONFIDENCE_THRESHOLD` if too many faces are missed
- Adjust `COVERED_STD_THRESHOLD` based on your specific use case

## 🎯 Use Cases

- **Security Systems**: Monitor mask compliance in buildings
- **Healthcare**: Ensure mask-wearing in medical facilities  
- **Retail**: Track mask usage in stores
- **Educational**: Demonstrate computer vision concepts
- **Research**: Analyze mask-wearing patterns in video data

## 🔮 Future Enhancements

- Replace texture-based classification with trained deep learning model
- Add support for multiple mask types (N95, cloth, surgical)
- Implement real-time camera feed processing
- Add data logging and analytics features
- Support for batch video processing

## 📝 Notes

- The current mask detection uses a simple texture analysis approach
- For production use, consider training a dedicated mask classification model
- The `MaskDetector` class provides a framework for integrating more sophisticated models
- Detection accuracy depends on video quality, lighting, and face angle

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!
