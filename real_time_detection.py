import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('deepfake_cnn_model.h5')

def predict_frame(frame):
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    prediction = model.predict(input_frame)[0][0]
    return prediction

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    prediction = predict_frame(frame)
    label = 'Fake' if prediction > 0.5 else 'Real'
    color = (0, 0, 255) if label == 'Fake' else (0, 255, 0)

    cv2.putText(frame, f'{label} ({prediction:.2f})', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Deepfake Real-Time Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
