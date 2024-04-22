# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Directorios
base_dir = "./"
test_dir = os.path.join(base_dir, "test")

# Generación de datos de prueba
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Cargar el modelo entrenado
model = load_model("lego_brick_classifier.keras")

# %%
# Evaluación del modelo
predictions = model.predict(test_generator) # Probabilidades de las clases
y_true = test_generator.classes
y_pred = np.argmax(predictions, axis=1)

# Matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

# Informe de clasificación
class_names = sorted(test_generator.class_indices.keys())
class_report = classification_report(y_true, y_pred, target_names=class_names)

# Visualización de la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %%
# Visualización de algunas imágenes con sus predicciones
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_generator[i][0][0])
    plt.title(f"True: {class_names[y_true[i]]}\nPredicted: {class_names[y_pred[i]]}", fontsize=9)
    plt.axis('off')

plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Para ajustar el espacio entre las imágenes
plt.show()

# Imprimir informe de clasificación
print("Classification Report:")
print(class_report)



