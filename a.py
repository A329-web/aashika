import cv2
import numpy as np

# Start webcam capture
cap = cv2.VideoCapture(0)

# Check if the webcam opens correctly
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# Initial filter mode
mode = "original"

print("Filter Controls:")
print("[o] Original | [r] Red | [g] Green | [b] Blue | [e] Edge Detection | [q] Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    if mode == "red":
        filtered = frame.copy()
        filtered[:, :, 0] = 0  # Remove Blue
        filtered[:, :, 1] = 0  # Remove Green
    elif mode == "green":
        filtered = frame.copy()
        filtered[:, :, 0] = 0  # Remove Blue
        filtered[:, :, 2] = 0  # Remove Red
    elif mode == "blue":
        filtered = frame.copy()
        filtered[:, :, 1] = 0  # Remove Green
        filtered[:, :, 2] = 0  # Remove Red
    elif mode == "edge":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered = cv2.Canny(gray, 80, 100)  # Apply Canny Edge Detection
    else:
        filtered = frame

    # Display the filtered output
    cv2.imshow("Webcam - Color Filters & Edge Detection", filtered)

    key = cv2.waitKey(1) & 0xFF

    # Change modes with key presses
    if key == ord("r"):
        mode = "red"
    elif key == ord("g"):
        mode = "green"
    elif key == ord("b"):
        mode = "blue"
    elif key == ord("e"):
        mode = "edge"
    elif key == ord("o"):
        mode = "original"
    elif key == ord("q"):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
