import cv2

import mediapipe as mp

import pyautogui

import numpy as np



mp_hands = mp.solutions.hands

mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

cam = cv2.VideoCapture(0)



STEERING_SENSITIVITY = 2.0   # smoother & more sensitive steering

DEAD_ZONE = 10

MAX_STEERING_ANGLE = 70



def calculate_steering_angle(landmarks, w, h):

    wrist = np.array([landmarks[0].x * w, landmarks[0].y * h])

    middle_mcp = np.array([landmarks[9].x * w, landmarks[9].y * h])

    direction_vector = middle_mcp - wrist

    angle = np.degrees(np.arctan2(direction_vector[0], -direction_vector[1]))

    angle = np.clip(angle * STEERING_SENSITIVITY, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

    return angle



def is_hand_open(landmarks):

    """Check if hand is open or closed based on finger tip y positions"""

    fingers = [8, 12, 16, 20]  # Tip landmarks for index, middle, ring, pinky

    open_fingers = 0

    for tip in fingers:

        if landmarks[tip].y < landmarks[tip - 2].y:  # tip above joint → open

            open_fingers += 1

    return open_fingers >= 3  # open if 3 or more fingers extended



print("Controls: 🖐 Open = Accelerate | ✊ Closed = Brake | Rotate = Steer | 'q' = Quit")



while True:

    success, frame = cam.read()

    if not success:

        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)



    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            angle = calculate_steering_angle(hand_landmarks.landmark, w, h)

            open_hand = is_hand_open(hand_landmarks.landmark)



            # Steering control

            if angle < -DEAD_ZONE:

                pyautogui.keyDown('left')

                pyautogui.keyUp('right')

                steering_text = "LEFT"

                color = (0, 0, 255)

            elif angle > DEAD_ZONE:

                pyautogui.keyDown('right')

                pyautogui.keyUp('left')

                steering_text = "RIGHT"

                color = (255, 0, 0)

            else:

                pyautogui.keyUp('left')

                pyautogui.keyUp('right')

                steering_text = "STRAIGHT"

                color = (0, 255, 0)



            # Acceleration / Brake control

            if open_hand:

                pyautogui.keyDown('up')

                pyautogui.keyUp('down')

                accel_text = "ACCELERATING 🟢"

                accel_color = (0, 255, 0)

            else:

                pyautogui.keyDown('down')

                pyautogui.keyUp('up')

                accel_text = "BRAKING 🔴"

                accel_color = (0, 0, 255)



            # Display info

            cv2.putText(frame, f"Steering: {steering_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(frame, accel_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, accel_color, 2)

            cv2.putText(frame, f"Angle: {angle:.1f}°", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)



    else:

        pyautogui.keyUp('up'); pyautogui.keyUp('down'); pyautogui.keyUp('left'); pyautogui.keyUp('right')

        cv2.putText(frame, "NO HAND DETECTED", (w//2-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)



    cv2.imshow("Car Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        pyautogui.keyUp('up'); pyautogui.keyUp('down'); pyautogui.keyUp('left'); pyautogui.keyUp('right')

        break



cam.release()

cv2.destroyAllWindows()