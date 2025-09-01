import cv2
import numpy as np
import os

# Load the pre-trained model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Open the RTSP stream
cap = cv2.VideoCapture("rtsp://admin:admin123@10.101.0.20:554/avstream/channel=2/stream=0.sdp", cv2.CAP_FFMPEG)

# Create a named window with normal (resizable) properties
cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

# Set desired display window size (e.g., 800x600)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
cv2.resizeWindow("Face Detection", DISPLAY_WIDTH, DISPLAY_HEIGHT)

CONFIDENCE_THRESHOLD = 0.5
COVERED_STD_THRESHOLD = 30  # Tune this based on your video

def is_face_masked(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    stddev = np.std(gray)
    return stddev < COVERED_STD_THRESHOLD

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            face_roi = frame[y1:y2, x1:x2]

            masked = False
            if face_roi.size != 0:
                masked = is_face_masked(face_roi)

            color = (0, 0, 255) if masked else (0, 255, 0)
            label = "UNMASKED" if masked else "MASKED"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Resize frame to fit display window size before showing
    resized_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("Face mask Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
