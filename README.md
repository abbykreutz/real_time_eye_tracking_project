# Real-Time Eye Tracking System

## Project Description

A real-time eye-tracking system that classifies eye states using OpenCV and MediaPipe Face Landmark detection. 
The system supports both eyes states independently and can detect winking. It measures blink count over time, blink frequency (blinks per minute), and blink duration.

### Features

- Independent left and right eye state classification (OPEN/CLOSED)
- Eye Aspect Ratio (EAR) computation to detect eye openness
- Blink frequency calculation in blinks per minute (BPM)
- Blink duration measurement in seconds
- Winking detection (asymmetric eye states)
- 60-second sliding window for blink metrics
- Live video visualization with color-coded eye status

## Installation Instructions

### Requirements

- Python 3.8 or higher
- Webcam or video input device
- macOS, Linux, or Windows

### Dependencies

Install required packages:

- opencv-python>=4.8.0
- mediapipe>=0.10.0
- numpy>=1.24.0
- scipy>=1.10.0

```bash
pip install opencv-python numpy mediapipe scipy
```

### Download Model

Download the MediaPipe Face Landmark model and place it in the project directory:

# The face_landmarker.task file should be in the same directory as eye_tracker.py

You can download the model from the MediaPipe GitHub repository or use the direct download link provided in the MediaPipe documentation.

## Usage Examples

### Basic Usage

Run the eye tracker with default settings:

```bash
python eye_tracker.py
```

The application will open a webcam window displaying:

- Left and right eye state (OPEN/CLOSED)
- Individual Eye Aspect Ratio (EAR) values for each eye
- Blink counts for each eye
- Blink frequency in blinks per minute (BPM)
- Blink duration in seconds
- Wink detection
- Face detection
- Frame counter

### Exiting
Press 'q' to quit the application.

### Adjusting EAR Threshold
To modify the sensitivity of eye-open detection, edit the threshold in eye_tracker.py:

```bash
EyeTracker(ear_threshold=0.21)  # Default: 0.21
```

Test threshold values between 0.18-0.25, find what works best!

## Known Limitations

- Requires good lighting conditions for accurate face detection
- Performs best when face is directly facing the camera (head not turned to side)
- Single face detection only (system handles one face per frame)
- Blink frequency is calculated over the time span between first and last detected blink, not total elapsed time
- Minimum blink duration is 2 frames at 30 FPS (~67 milliseconds) to avoid false positives
- Performance depends on camera quality and frame rate
- May have difficulty with extreme facial angles or expressions
- Tinted sunglasses or occlusions can affect landmark detection accuracy, although no issues have been detected with regulaer glasses or regular sunglasses outdoors
- Have not run into personal issues with testing where webcam lags or crashes