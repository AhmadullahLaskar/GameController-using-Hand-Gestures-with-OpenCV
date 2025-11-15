import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Initialize webcam
cam = cv2.VideoCapture(0)

# Game controller settings
ACTION_COOLDOWN = 0.001  # Reduced cooldown for faster response
last_action_time = 0
sensitivity = 5  # Increased sensitivity

print("🎮 ADVANCED GAME CONTROLLER")
print("← → ↑ ↓ Control with hand movements")
print("Quick gestures for instant response")
print("Press 'q' to quit")

while True:
    success, frame = cam.read()
    if not success:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw control zones
    cv2.rectangle(frame, (0, 0), (w//3, h), (50, 50, 200), 2)  # Left zone
    cv2.rectangle(frame, (2*w//3, 0), (w, h), (50, 200, 50), 2)  # Right zone
    cv2.rectangle(frame, (0, 0), (w, h//4), (200, 200, 50), 2)  # Up zone
    cv2.rectangle(frame, (0, 3*h//4), (w, h), (200, 50, 50), 2)  # Down zone

    current_time = time.time()
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of index fingertip
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Draw fingertip and path
            cv2.circle(frame, (x, y), 15, (0, 255, 255), cv2.FILLED)
            cv2.circle(frame, (x, y), 20, (0, 255, 0), 2)

            # Check cooldown
            if current_time - last_action_time > ACTION_COOLDOWN:
                
                # LEFT - More sensitive zone
                if x < w // 3:
                    pyautogui.press('left')
                    cv2.putText(frame, 'LEFT', (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    last_action_time = current_time
                    print("LEFT triggered")

                # RIGHT - More sensitive zone  
                elif x > 2 * w // 3:
                    pyautogui.press('right')
                    cv2.putText(frame, 'RIGHT', (w-200, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    last_action_time = current_time
                    print("RIGHT triggered")

                # UP - Smaller zone for accidental trigger prevention
                elif y < h // 4:
                    pyautogui.press('up')
                    cv2.putText(frame, 'JUMP', (w//2-100, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
                    last_action_time = current_time
                    print("JUMP triggered")

                # DOWN - Smaller zone for accidental trigger prevention
                elif y > 3 * h // 4:
                    pyautogui.press('down') 
                    cv2.putText(frame, 'SLIDE', (w//2-100, h-50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255), 3)
                    last_action_time = current_time
                    print("SLIDE triggered")

    # Display status
    time_since_last_action = current_time - last_action_time
    cooldown_status = "READY" if time_since_last_action > ACTION_COOLDOWN else f"COOLDOWN: {ACTION_COOLDOWN-time_since_last_action:.1f}s"
    cv2.putText(frame, f"Status: {cooldown_status}", (10, h-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display window
    cv2.imshow("Advanced Game Controller - Subway Surfers/Temple Run", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()