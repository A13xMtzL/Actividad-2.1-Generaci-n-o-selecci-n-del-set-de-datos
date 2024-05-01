
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Directorios
base_dir = "A:/Escuela/Octavos_Semestre/M2_IA/Act_2.1/new__/"
test_dir = os.path.join(base_dir, "A:/Escuela/Octavos_Semestre/M2_IA/Act_2.1/new__/_test")
train_dir = os.path.join(base_dir, "A:/Escuela/Octavos_Semestre/M2_IA/Act_2.1/new__/_train")

# Generación de datos de prueba
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

train_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

# Cargar el modelo entrenado
model = load_model("__model_best_new_2.keras")


# # Matriz de Confusión de Test


# Evaluación del modelo
predictions = model.predict(test_generator) # Probabilidades de las clases
y_true = test_generator.classes
y_pred = np.argmax(predictions, axis=1)

# Matriz de confusión
conf_matrix_test = confusion_matrix(y_true, y_pred)

# Informe de clasificación
class_names_test = sorted(test_generator.class_indices.keys())
class_report_test = classification_report(y_true, y_pred, target_names=class_names_test)

# Visualización de la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="GnBu", xticklabels=class_names_test, yticklabels=class_names_test)
plt.title("Confusion Matrix Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print("Classification Report:")
print(class_report_test)


# Evaluación del modelo
predictions = model.predict(train_generator) # Probabilidades de las clases
y_true = train_generator.classes
y_pred = np.argmax(predictions, axis=1)

# Matriz de confusión
conf_matrix_train = confusion_matrix(y_true, y_pred)

# Informe de clasificación
class_names_train = sorted(train_generator.class_indices.keys())
class_report_train = classification_report(y_true, y_pred, target_names=class_names_train)

# Visualización de la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_train, annot=True, fmt="d", cmap="RdPu", xticklabels=class_names_train, yticklabels=class_names_train)
plt.title("Confusion Matrix Train Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print("Classification Report:")
print(class_report_train)

# Obtén las primeras 9 imágenes y etiquetas del generador de pruebas
images, labels = next(test_generator)

# Haz predicciones para estas imágenes
predictions = model.predict(images)

# Convierte las predicciones a clases
predicted_classes = np.argmax(predictions, axis=1)

# Convierte las etiquetas verdaderas a clases
true_classes = np.argmax(labels, axis=1)

# Imprime las primeras 9 predicciones y las etiquetas verdaderas
for i in range(64):
    print(f"Image {i+1}:")  # Podemos cambiar el rango para ver más imágenes
    print(f"Predicted class:\t {class_names_test[predicted_classes[i]]}")
    print(f"True class:\t {class_names_test[true_classes[i]]}")
    print()

# Visualizamos algunas imágenes con sus predicciones
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i +1])
    plt.title(f"True: {class_names_test[true_classes[i + 10]]}\nPredicted: {class_names_test[predicted_classes[i + 10]]}", fontsize=9)
    plt.axis('off')
    
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Para ajustar el espacio entre las imágenes
plt.show()
    
print("Classification Report:")
print(class_report_test)