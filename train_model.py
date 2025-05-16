import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parámetros generales
IMG_SIZE = 64
IMAGE_DIR = 'dataset'
EPOCHS = 50
BATCH_SIZE = 32

def get_label_from_filename(filename):
    return filename[0].upper()

def load_images_and_labels():
    data, labels = [], []
    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith(".jpg"):
            path = os.path.join(IMAGE_DIR, filename)
            try:
                img = cv2.imread(path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img / 255.0
                data.append(img)
                labels.append(get_label_from_filename(filename))
            except Exception as e:
                print(f"❌ Error con {filename}: {e}")
    return np.array(data), np.array(labels)

# Cargar y procesar datos
X, y = load_images_and_labels()
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y_cat, random_state=42)

# Aumentación de datos más agresiva
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1
)
datagen.fit(X_train)

# Calcular pesos para clases desbalanceadas
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(enumerate(class_weights))

# Modelo CNN básico optimizado
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrenamiento
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(patience=5)],
    class_weight=class_weight_dict
)

# Guardar modelo
os.makedirs('model', exist_ok=True)
model.save('model/sign_model.h5')
np.save('model/labels.npy', le.classes_)

# Resultados de entrenamiento
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
plt.savefig('model/training_metrics.png')
plt.show()

# Predicción y métricas
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\n📊 Reporte de clasificación:")
report = classification_report(y_true_labels, y_pred_labels, target_names=le.classes_)
print(report)

# Guardar reporte
with open("model/classification_report.txt", "w") as f:
    f.write(report)

# Matriz de confusión
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png')
plt.show()

# Precisión por clase
precision_per_class = precision_score(y_true_labels, y_pred_labels, average=None)
plt.figure(figsize=(10, 5))
sns.barplot(x=le.classes_, y=precision_per_class)
plt.title("Precisión por letra")
plt.xlabel("Letra")
plt.ylabel("Precisión")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('model/precision_per_letter.png')
plt.show()
