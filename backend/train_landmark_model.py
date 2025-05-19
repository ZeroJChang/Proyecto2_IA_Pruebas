# train_landmark_model.py
import os
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_DIR = "dataset_landmarks"
X = []
y = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        label = file.split(".")[0].upper()
        path = os.path.join(DATA_DIR, file)
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                X.append([float(val) for val in row])
                y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Total muestras: {len(X)}")

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=1)
clf.fit(X_train, y_train)

# Evaluar
y_pred = clf.predict(X_test)
print("🔍 Evaluación:")
print(classification_report(y_test, y_pred))
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Guardar modelo
joblib.dump(clf, "../model/landmark_model.pkl")
np.save("../model/landmark_labels.npy", np.unique(y))
print("✅ Modelo y etiquetas guardados.")
