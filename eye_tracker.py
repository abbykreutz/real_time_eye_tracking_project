import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

# Load the Face Landmark Model
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "face_landmarker.task"

# # --- TEST OPENCV INSTALLATION ---

# print("OpenCV:", cv.__version__)
# img = np.zeros((120, 400, 3), dtype=np.uint8)
# cv.putText(img, "OpenCV OK", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
# # If you installed a non-headless build, you can display a window:
# cv.imshow("hello", img); cv.waitKey(0)
# # Always safe (headless or not): save to file
# cv.imwrite("hello.png", img)

# --- STEP 1: WORKING WEBCAM DISPLAY ---
def main():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.\n"
              "On macOS: System Settings → Privacy & Security → Camera → allow Terminal/VS Code.")
        return
    
    face_mesh = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = vision.FaceLandmarker.create_from_options(face_mesh)
    
    verified = False

    while True:
        success, frame = cap.read()
        if not success:
            break
    
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(cv.getTickCount() / cv.getTickFrequency() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Verify MediaPipe once by processing single frame
        if not verified:
            if result.face_landmarks:
                print("MediaPipe FaceMesh: OK - face detected")
            else:
                print("MediaPipe FaceMesh: OK - no face detected (handled)")
            verified = True

        cv.imshow("Real-Time Eye Tracking Webcam (press q to quit)", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()

# # --- CODE ARCHITECTURE ---
# class EyeTracker:
#     def __init__(self):
#         # Initialize MediaPipe Face Mesh
#         # Set EAR threshold
        
#     def calculate_ear(self, eye_landmarks):
#         # Compute Eye Aspect Ratio
#         # Return float value
        
#     def get_eye_landmarks(self, landmarks, indices, frame_w, frame_h):
#         # Extract specific eye landmarks
#         # Return list of (x, y) coordinates
        
#     def process_frame(self, frame):
#         # Process single frame
#         # Return annotated frame
        
#     def run(self):
#         # Main loop: capture, process, display
#         # Handle keyboard input