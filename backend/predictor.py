import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 64
MODEL_PATH = 'model/sign_model.h5'
LABELS_PATH = 'model/labels.npy'

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]
    predicted_label = labels[class_index]

    return predicted_label, confidence

if __name__ == "__main__":
    test_path = "images/B0_jpg.rf.aaaa...jpg" 
    label, conf = predict_image(test_path)
    print(f"Predicción: {label} (confianza: {conf:.2f})")
