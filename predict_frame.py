import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('deepfake_cnn_model.keras')

# Provide correct image path
img_path = 'frames/fake_frames/0.jpg'  # Update this to your actual image file

# Read and preprocess image
img = cv2.imread(img_path)
if img is None:
    print(f"Error: Unable to load image at {img_path}")
else:
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        print("Predicted: Fake")
    else:
        print("Predicted: Real")
