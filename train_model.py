import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Configuración
IMG_SIZE = 128
IMAGE_DIR = 'dataset'
EPOCHS = 50
BATCH_SIZE = 32
USE_MEDIAPIPE = False  # Cambiar a True si quieres usar detección de manos (más lento)

def simple_preprocess(image):
    """Preprocesamiento rápido sin MediaPipe"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    resized = cv2.resize(equalized, (IMG_SIZE, IMG_SIZE))
    return resized / 255.0

def load_images_and_labels():
    """Carga imágenes con opción para usar MediaPipe"""
    data, labels = [], []
    
    print("Procesando imágenes...")
    total_files = len([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
    processed = 0
    
    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(IMAGE_DIR, filename)
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                
                if USE_MEDIAPIPE:
                    # Preprocesamiento con MediaPipe (más preciso pero lento)
                    processed_img = preprocess_with_mediapipe(img)
                else:
                    # Preprocesamiento rápido
                    processed_img = simple_preprocess(img)
                
                if processed_img is not None:
                    data.append(processed_img)
                    labels.append(filename[0].upper())  # Asume que el primer carácter es la letra
                
                processed += 1
                if processed % 100 == 0:
                    print(f"Procesadas {processed}/{total_files} imágenes...")
                    
            except Exception as e:
                print(f"Error con {filename}: {str(e)[:100]}...")
    
    return np.array(data), np.array(labels)

# Configuración de aumento de datos
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def create_model(input_shape, num_classes):
    """Crea el modelo CNN"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def main():
    # Cargar datos
    X, y = load_images_and_labels()
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Añadir canal
    
    # Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded)
    
    # Balanceo de clases
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, stratify=y_cat, random_state=42)
    
    # Crear y entrenar modelo
    model = create_model((IMG_SIZE, IMG_SIZE, 1), len(le.classes_))
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6)
        ],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Guardar modelo
    os.makedirs('model', exist_ok=True)
    model.save('model/sign_model.h5')
    np.save('model/labels.npy', le.classes_)
    
        # Evaluación del modelo
    print("\nEvaluando modelo con el conjunto de prueba...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión en prueba: {test_acc:.4f}")
    print(f"Pérdida en prueba: {test_loss:.4f}")

    # Generar predicciones
    y_pred = model.predict(X_test, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # Reporte de clasificación detallado
    print("\n📊 Reporte de Clasificación Detallado:")
    report = classification_report(
        y_true_labels, 
        y_pred_labels, 
        target_names=le.classes_, 
        zero_division=1,
        digits=4
    )
    print(report)

    # Guardar reporte en archivo
    with open("model/classification_report.txt", "w") as f:
        f.write(f"Precisión en prueba: {test_acc:.4f}\n")
        f.write(f"Pérdida en prueba: {test_loss:.4f}\n\n")
        f.write(report)

    # Matriz de confusión
    plt.figure(figsize=(15, 12))
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=le.classes_, 
        yticklabels=le.classes_,
        cbar=False,
        annot_kws={"size": 8}
    )
    plt.title('Matriz de Confusión', fontsize=16, pad=20)
    plt.xlabel('Predicciones', fontsize=14)
    plt.ylabel('Valores Reales', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico de precisión por clase
    precision_per_class = cm.diagonal() / cm.sum(axis=0)
    recall_per_class = cm.diagonal() / cm.sum(axis=1)
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)

    plt.figure(figsize=(18, 6))
    
    # Precisión por clase
    plt.subplot(1, 3, 1)
    sns.barplot(x=le.classes_, y=precision_per_class, palette='viridis')
    plt.title('Precisión por Clase', fontsize=14)
    plt.xlabel('Letra', fontsize=12)
    plt.ylabel('Precisión', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=90)
    for i, v in enumerate(precision_per_class):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)

    # Recall por clase
    plt.subplot(1, 3, 2)
    sns.barplot(x=le.classes_, y=recall_per_class, palette='viridis')
    plt.title('Recall por Clase', fontsize=14)
    plt.xlabel('Letra', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=90)
    for i, v in enumerate(recall_per_class):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)

    # F1-Score por clase
    plt.subplot(1, 3, 3)
    sns.barplot(x=le.classes_, y=f1_per_class, palette='viridis')
    plt.title('F1-Score por Clase', fontsize=14)
    plt.xlabel('Letra', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=90)
    for i, v in enumerate(f1_per_class):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('model/class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Gráficas de entrenamiento
    plt.figure(figsize=(18, 6))
    
    # Precisión durante entrenamiento
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión durante Entrenamiento', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Precisión', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Pérdida durante entrenamiento
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida durante Entrenamiento', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('model/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico de las peores predicciones
    worst_predictions = []
    for i in range(len(X_test)):
        if y_pred_labels[i] != y_true_labels[i]:
            confidence = y_pred[i][y_pred_labels[i]]
            worst_predictions.append((i, confidence))
    
    worst_predictions = sorted(worst_predictions, key=lambda x: x[1], reverse=True)[:10]

    plt.figure(figsize=(15, 8))
    for idx, (i, confidence) in enumerate(worst_predictions):
        plt.subplot(2, 5, idx + 1)
        plt.imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
        plt.title(f"Real: {le.classes_[y_true_labels[i]]}\nPred: {le.classes_[y_pred_labels[i]]}\nConf: {confidence:.2f}")
        plt.axis('off')
    
    plt.suptitle('Peores Predicciones (mayor confianza incorrecta)', fontsize=16)
    plt.tight_layout()
    plt.savefig('model/worst_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ Evaluación completada y gráficas guardadas en la carpeta 'model'")
    
if __name__ == "__main__":
    main()