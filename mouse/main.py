import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

clicking = False
print("🖐 Hand Gesture Mouse Controller (Thumb Up = Click)")
print("👉 Move Index Finger = Move Cursor | 👍 = Click | ✊ = Stop | 'q' = Quit")

def is_thumb_up(lm):
    """Check if thumb is pointing up."""
    return lm[4].y < lm[3].y < lm[2].y  # y decreases upward

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    success, frame = cam.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            index_finger = (lm[8].x * w, lm[8].y * h)

            # Convert to screen coordinates
            screen_x = np.interp(index_finger[0], [0, w], [0, screen_w])
            screen_y = np.interp(index_finger[1], [0, h], [0, screen_h])
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)

            # Draw index point
            cv2.circle(frame, (int(index_finger[0]), int(index_finger[1])), 10, (0, 255, 0), -1)

            # Check for thumb up gesture
            if is_thumb_up(lm):
                if not clicking:
                    clicking = True
                    pyautogui.click()
                    cv2.putText(frame, "👍 CLICK!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                clicking = False

            # Detect Fist (all fingers folded)
            fingers_folded = 0
            for tip in [8, 12, 16, 20]:
                if lm[tip].y > lm[tip - 2].y:
                    fingers_folded += 1
            if fingers_folded == 4:
                cv2.putText(frame, "✊ Fist Detected - Stop", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                continue

    cv2.putText(frame, "Move: Index | Click: Thumb Up | Stop: Fist | Quit: q",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("Hand Mouse Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
q