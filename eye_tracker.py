# --- STEP 1: ENVIRONMENT SETUP ---
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

# Load Face Landmark Model
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "face_landmarker.task"

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# # TEST OPENCV INSTALLATION
# print("OpenCV:", cv.__version__)
# img = np.zeros((120, 400, 3), dtype=np.uint8)
# cv.putText(img, "OpenCV OK", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
# # If you installed a non-headless build, you can display a window:
# cv.imshow("hello", img); cv.waitKey(0)
# # Always safe (headless or not): save to file
# cv.imwrite("hello.png", img)

# --- CODE ARCHITECTURE ---
class EyeTracker:
    def __init__(self, model_path: str = MODEL_PATH, ear_threshold: float = 0.21):
        face_mesh = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(face_mesh)
        self.ear_threshold = ear_threshold
        self.frame_count = 0
        self._face_detected = None

    def euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
        
    def calculate_ear(self, eye_landmarks):

        p1, p2, p3, p4, p5, p6 = eye_landmarks

        v1 = self.euclidean(p2, p6)
        v2 = self.euclidean(p3, p5)
        h = self.euclidean(p1, p4)

        if h == 0:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
        
    def get_eye_landmarks(self, landmarks, indices, frame_w, frame_h):
        eye_landmarks = []
        for eye in indices:
            x = int(landmarks[eye].x * frame_w)
            y = int(landmarks[eye].y * frame_h)
            eye_landmarks.append((x, y))
        return eye_landmarks
        
    def process_frame(self, frame):
        self.frame_count += 1

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(cv.getTickCount() / cv.getTickFrequency() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        detected = bool(result.face_landmarks)

        if detected != self._face_detected:
            print("Face Detected" if detected else "No Face Detected")
            self._face_detected = detected

        h, w = frame.shape[:2]

        if detected:

            left_eye = self.get_eye_landmarks(result.face_landmarks[0], LEFT_EYE_INDICES, w, h)
            right_eye = self.get_eye_landmarks(result.face_landmarks[0], RIGHT_EYE_INDICES, w, h)

            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            cv.putText(frame, f"EAR: {avg_ear:.3f}", (10,90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            eye_color = (255, 0, 0)  # blue (OPEN)
            eye_state = "OPEN"
            if avg_ear < self.ear_threshold:
                eye_color = (0, 0, 255)  # red (CLOSED)
                eye_state = "CLOSED"

            cv.polylines(frame, [np.array(left_eye, dtype=np.int32)], isClosed=True, color=eye_color, thickness=2)
            cv.polylines(frame, [np.array(right_eye, dtype=np.int32)], isClosed=True, color=eye_color, thickness=2)

            status_text = f"EYE STATUS ({eye_state})"
            color = (0, 255, 0)
        else:
            status_text = "FACE NOT DETECTED"
            color = (0, 0, 255)

        # Process single frame
        # Return annotated frame
        cv.putText(frame, status_text, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 0) if result.face_landmarks else (0, 0, 255), 2)

        cv.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame

    # # --- STEP 2: FACE DETECTION ---
    # def _draw_all_landmarks(self, frame_bgr, landmarks, radius: int = 1):
    #     """Draws every facial landmark point."""
    #     h, w = frame_bgr.shape[:2]
    #     for lm in landmarks:
    #         x = int(lm.x * w)
    #         y = int(lm.y * h)
    #         if 0 <= x < w and 0 <= y < h:
    #             cv.circle(frame_bgr, (x, y), radius, (0, 255, 0), -1)
        
    def run(self):
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FPS, 30)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open webcam.\n"
                "On macOS: System Settings → Privacy & Security → Camera → allow Terminal/VS Code.")
            return

        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame = self.process_frame(frame)

            cv.imshow("Real-Time Eye Tracking Webcam (press q to quit)", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    EyeTracker().run()