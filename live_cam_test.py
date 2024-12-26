import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the mask detection model
mask_model = load_model('C:/Users/M-Tech/Desktop/github/face_mask/model/face_mask_detection_model.h5')

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))  # Resize to model input size
        face_array = img_to_array(face_resized) / 255.0  # Normalize pixel values
        face_array = np.expand_dims(face_array, axis=0)

        # Predict using the mask detection model
        probabilities = mask_model.predict(face_array)[0][0]
        wearing_mask = (1 - probabilities) * 100
        not_wearing_mask = probabilities * 100

        # Annotate the frame with results
        label = "Mask" if probabilities < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        confidence = wearing_mask if label == "Mask" else not_wearing_mask
        text = f"{label}: {confidence:.2f}%"

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the resulting frame
    cv2.imshow('Face Mask Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
