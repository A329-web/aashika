import cv2
import mediapipe as mpq
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from math import hypot
import screen_brightness_control as sb

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    volume_range = volume.GetVolumeRange()
    min_vol = volume_range[0]
    max_vol = volume_range[1]
except Exception as e:
    print(f"Error initializing Pycaw: {e}")
    exit()


cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    if not success:
        break
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lm_list = []
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

     
        if len(lm_list) >= 9:
            x1, y1 = lm_list[4][1], lm_list[4][2]
            x2, y2 = lm_list[8][1], lm_list[8][2]

        
            cv2.circle(img, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            length = hypot(x2 - x1, y2 - y1)

     
            vol = np.interp(length, [15, 150], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

      
            bright = np.interp(length, [15, 150], [0, 100])
            try:
                sb.set_brightness(int(bright))
            except Exception as e:
                print(f"Brightness Error: {e}")

         
            cv2.putText(
                img,
                f"Len: {int(length)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

    cv2.imshow("Gesture Volume & Brightness Control", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
