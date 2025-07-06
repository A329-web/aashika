import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

# Initial position of the object (e.g., a circle)
x_pos, y_pos = 250, 250

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (hand)
        hand = max(contours, key=cv2.contourArea)

        # Get center of the bounding box
        x, y, w, h = cv2.boundingRect(hand)
        cx = x + w // 2
        cy = y + h // 2

        # Draw center point on hand
        cv2.circle(frame_copy, (cx, cy), 10, (0, 0, 255),-1)

        # Move the object (circle) toward the hand
        x_pos = int(0.8 * x_pos + 0.2 * cx)
        y_pos = int(0.8 * y_pos + 0.2 * cy)

    # Draw controlled circle
    cv2.circle(frame_copy, (x_pos, y_pos), 30, (0, 255, 0), -1)

    # Show the result
    cv2.imshow("Gesture Control", frame_copy)
    cv2.imshow("Mask", mask)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release and cleanup
cap.release()
cv2.destroyAllWindows()
