# --- STEP 1: ENVIRONMENT SETUP ---
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from collections import deque

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
        self.fps = 30

        # --- Lvl 1 Stretch Goal: Independent eye tracking (winking) ---
        
        # Left eye state tracking
        self.left_eye_state = "OPEN"
        self.left_ear_value = 0.0
        self.left_blink_count = 0
        self.left_blink_start_frame = None
        self.left_blink_duration = 0
        self.left_is_winking = False

        # Right eye state tracking
        self.right_eye_state = "OPEN"
        self.right_ear_value = 0.0
        self.right_blink_count = 0
        self.right_blink_start_frame = None
        self.right_blink_duration = 0
        self.right_is_winking = False

        # Blink frequency
        self.max_blink_frames = self.fps * 60
        self.left_blink_timestamps = deque(maxlen=self.max_blink_frames)
        self.right_blink_timestamps = deque(maxlen=self.max_blink_frames)
        self.min_blink_frames = 2


    def euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
        
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR) for a each eye."""
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
    
    def update_eye_state(self, eye, ear_value, current_frame):
        """Update eye state (OPEN/CLOSED) and count blinks based on EAR threshold."""
        if eye == "LEFT":
            prev_state = self.left_eye_state
            self.left_ear_value = ear_value

            if ear_value < self.ear_threshold:
                self.left_eye_state = "CLOSED"
            else:
                self.left_eye_state = "OPEN"

            if prev_state == "OPEN" and self.left_eye_state == "CLOSED":
                self.left_blink_start_frame = current_frame

            if prev_state == "CLOSED" and self.left_eye_state == "OPEN":
                if self.left_blink_start_frame is not None:
                    blink_duration_frame = current_frame - self.left_blink_start_frame
                    if blink_duration_frame >= self.min_blink_frames:
                        self.left_blink_count += 1
                        self.left_blink_duration = blink_duration_frame
                        self.left_blink_timestamps.append(current_frame)
                self.left_blink_start_frame = None

        elif eye == "RIGHT":
            prev_state = self.right_eye_state
            self.right_ear_value = ear_value

            if ear_value < self.ear_threshold:
                self.right_eye_state = "CLOSED"
            else:
                self.right_eye_state = "OPEN"

            if prev_state == "OPEN" and self.right_eye_state == "CLOSED":
                self.right_blink_start_frame = current_frame

            if prev_state == "CLOSED" and self.right_eye_state == "OPEN":
                if self.right_blink_start_frame is not None:
                    blink_duration_frame = current_frame - self.right_blink_start_frame
                    if blink_duration_frame >= self.min_blink_frames:
                        self.right_blink_count += 1
                        self.right_blink_duration = blink_duration_frame
                        self.right_blink_timestamps.append(current_frame)
                self.right_blink_start_frame = None

    def detect_winking(self):
        """Detect winking by tracking asymmetric eye states (one eye open, the other closed)."""
        self.left_is_winking = (self.left_eye_state == "CLOSED" and self.right_eye_state == "OPEN")
        self.right_is_winking = (self.left_eye_state == "OPEN" and self.right_eye_state == "CLOSED")

    def calculate_blink_frequency(self, blink_timestamps):
        """Calculate blinks per minute from timestamps"""
        if len(blink_timestamps) == 0:
            return 0.0
        
        if len(blink_timestamps) <= 1:
            return 0.0
        
        time_span_frames = blink_timestamps[-1] - blink_timestamps[0]
        time_span_seconds = time_span_frames / self.fps
        
        if time_span_seconds == 0:
            return 0.0
        
        blinks_per_minute = (len(blink_timestamps) / time_span_seconds) * 60
        return blinks_per_minute

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

            self.update_eye_state("LEFT", left_ear, self.frame_count)
            self.update_eye_state("RIGHT", right_ear, self.frame_count)

            self.detect_winking()

            left_bpm = self.calculate_blink_frequency(self.left_blink_timestamps)
            right_bpm = self.calculate_blink_frequency(self.right_blink_timestamps)

            left_eye_color = (0, 0, 255) if self.left_eye_state == "CLOSED" else (0, 225, 0)
            right_eye_color = (0, 0, 255) if self.right_eye_state == "CLOSED" else (0, 225, 0)

            cv.polylines(frame, [np.array(left_eye, dtype=np.int32)], isClosed=True, color=left_eye_color)
            cv.polylines(frame, [np.array(right_eye, dtype=np.int32)], isClosed=True, color=right_eye_color)

            left_text_color = (0, 0, 255) if self.left_eye_state == "CLOSED" else (0, 255, 0)
            cv.putText(frame, f"LEFT: {self.left_eye_state} (EAR:{self.left_ear_value:.3f})", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, left_text_color, 2)
            cv.putText(frame, f"Left Blinks {self.left_blink_count} | BPM: {left_bpm:.1f}", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv.putText(frame, f"L-Duration: {self.left_blink_duration / self.fps:.2f}s", (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            right_text_color = (0, 0, 255) if self.right_eye_state == "CLOSED" else (0, 255, 0)
            cv.putText(frame, f"RIGHT: {self.right_eye_state} (EAR:{self.right_ear_value:.3f})", (10, 120),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, right_text_color, 2)
            cv.putText(frame, f"Right Blinks {self.right_blink_count} | BPM: {right_bpm:.1f}", (10, 150),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv.putText(frame, f"R-Duration: {self.right_blink_duration / self.fps:.2f}s", (10, 180),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            wink_status = ""
            if self.left_is_winking:
                wink_status = "LEFT WINKING"
            elif self.right_is_winking:
                wink_status = "RIGHT WINKING"

            if wink_status:
                cv.putText(frame, wink_status, (10, 210),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            face_status = "FACE DETECTED"
            color = (0, 255, 0)
        else:
            face_status = "FACE NOT DETECTED"
            color = (0, 0, 255)

        cv.putText(frame, face_status, (w - 250, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv.putText(frame, f"Frame: {self.frame_count}", (w - 200, 60),
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