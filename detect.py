import cv2
import joblib
import numpy as np
from src.data_preprocessing import preprocess_data

# Load the pre-trained Random Forest model
model = joblib.load('models/random_forest_model.pkl')

# Load label map (ensure the order of labels is consistent with training)
#label_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}  # Adjust this map based on your training labels
label_map = {'angry': 'angry', 'disgust': 'disgust', 'fear': 'fear', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad', 'surprise': 'surprise'}

def predict_expression(face_img):
    face_img = cv2.resize(face_img, (48, 48))  # Resize to match training data
    face_img = preprocess_data(np.array([face_img]))  # Preprocess and reshape
    prediction = model.predict(face_img)
    print(f"Model prediction: {prediction[0]}")
    return label_map[prediction[0]]

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV's built-in Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray[y:y+h, x:x+w]
        
        # Predict the expression
        expression = predict_expression(face)
        
        # Draw a rectangle around the face and label it with the predicted expression
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, expression, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the frame with the predictions
    cv2.imshow('Facial Expression Recognition', frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
