import cv2
import numpy as np
import time
from datetime import datetime

# Initialize background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

# Initialize face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Set video resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('motion_tracking_output.avi', fourcc, 20.0, (640, 480))

# Motion tracking settings
frame_skip = 2
motion_threshold = 500
frame_counter = 0
last_motion_time = None

print("Press 'q' to quit, 'u' to increase motion sensitivity, 'd' to decrease.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    if frame_counter % frame_skip != 0:
        continue

    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(frame_resized, (5, 5), 0)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Detect motion
    fgmask = fgbg.apply(blurred_frame)
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > motion_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

    for (x, y, w, h) in faces:
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if motion_detected:
        if last_motion_time is None or (time.time() - last_motion_time) > 5:
            last_motion_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            cv2.imwrite(f"motion_detected_{timestamp}.jpg", frame_resized)
            print(f"Motion detected at {timestamp}")

        out.write(frame_resized)

    cv2.imshow("Motion and Face Tracking", frame_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('u'):
        motion_threshold += 50
        print(f"Increased motion sensitivity to {motion_threshold}")
    elif key == ord('d'):
        motion_threshold = max(100, motion_threshold - 50)
        print(f"Decreased motion sensitivity to {motion_threshold}")

cap.release()
out.release()
cv2.destroyAllWindows()
