import cv2
import sys
sys.stdout.reconfigure(encoding='utf-8')
import mediapipe as mp
import time
import numpy as np
import pyttsx3

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Filters
filters = [None, "GRAYSCALE", "SEPIA", "NEGATIVE", "BLUR", "CARTOON", "EDGES"]
current_filter = 0
last_switch_time = 0
switch_delay = 1  # seconds


# Filter application
def apply_filter(frame, filter_type):
    if filter_type == "GRAYSCALE":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_type == "SEPIA":
        kernel = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
        )
        return cv2.transform(frame, kernel)
    elif filter_type == "NEGATIVE":
        return cv2.bitwise_not(frame)
    elif filter_type == "BLUR":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    elif filter_type == "CARTOON":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(frame, 9, 250, 250)
        return cv2.bitwise_and(color, color, mask=edges)
    elif filter_type == "EDGES":
        return cv2.Canny(frame, 100, 200)
    else:
        return frame


# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

print(
    "🖐️ Show 5 fingers to switch filter | ✌️ Show 2 fingers to take screenshot | ESC to exit"
)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers
            lm = hand_landmarks.landmark
            fingers_up = [
                lm[8].y < lm[6].y,  # Index
                lm[12].y < lm[10].y,  # Middle
                lm[16].y < lm[14].y,  # Ring
                lm[20].y < lm[18].y,  # Pinky
            ]
            thumb_up = lm[4].x > lm[2].x
            finger_count = sum(fingers_up) + thumb_up

            # Gesture: Switch Filter (5 fingers)
            if finger_count == 5 and (time.time() - last_switch_time) > switch_delay:
                current_filter = (current_filter + 1) % len(filters)
                filter_name = filters[current_filter] or "No Filter"
                print(f"Switched to: {filter_name}")
                engine.say(f"Switched to {filter_name}")
                engine.runAndWait()
                last_switch_time = time.time()

            # Gesture: Take Screenshot (2 fingers)
            if finger_count == 2:
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"🖼️ Screenshot saved as {filename}")
                engine.say("Screenshot taken")
                engine.runAndWait()
                time.sleep(1)

    # Apply current filter
    filtered = apply_filter(frame.copy(), filters[current_filter])
    if filters[current_filter] in ["GRAYSCALE", "EDGES"]:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    # Show filter name
    cv2.putText(
        filtered,
        f"Filter: {filters[current_filter] or 'None'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )

    # Display result
    cv2.imshow("Gesture Photo App", filtered)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
