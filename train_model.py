import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Parámetros
IMG_SIZE = 64
IMAGE_DIR = 'dataset'  # Cambia si tu carpeta se llama diferente

def get_label_from_filename(filename):
    return filename[0].upper()  # Toma la letra inicial del nombre

def load_images_and_labels():
    data = []
    labels = []
    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith(".jpg"):
            path = os.path.join(IMAGE_DIR, filename)
            try:
                img = cv2.imread(path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                data.append(img)
                labels.append(get_label_from_filename(filename))
            except Exception as e:
                print(f"❌ Error con imagen {filename}: {e}")
    return np.array(data), np.array(labels)

# Cargar datos
X, y = load_images_and_labels()
X = X / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Dividir conjunto
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Aumentación de datos
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# Crear modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrenamiento
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(patience=3)]
)

# Crear carpeta si no existe
os.makedirs('model', exist_ok=True)
model.save('model/sign_model.h5')
np.save('model/labels.npy', le.classes_)

print("✅ Modelo entrenado y guardado en carpeta 'model'")

# Graficar resultados
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
