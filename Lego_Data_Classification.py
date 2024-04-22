# %%
# from google.colab import drive

# drive.mount("/content/drive")

# %%
# %cd "/content/drive/MyDrive/Act2.1"
# !dir

# %% [markdown]
# # LEGO Data Classification

# %% [markdown]
# ## Reescalado de imágenes y augmentation 

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


base_dir = "./"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# %%
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=100,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
)
# plt.figure()
# #subplot(r,c) provide the no. of rows and columns
# f, axarr = plt.subplots(1, 5, figsize=(30, 8))

# for i in range(5) :
#   axarr[i].imshow(train_datagen[0][0][0])

# %%
# path = "./"


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=25,
    class_mode="categorical",
    save_to_dir=base_dir+'/augmented/',
    save_prefix='aug',
    save_format='png'
)

images , labels = train_generator[0]

print(images.shape)
print(labels)


plt.figure()
#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(1, images.shape[0], figsize=(30, 4))

for i in range(images.shape[0]) :
  axarr[i].imshow(images[i])

# %%
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=25,
    class_mode="categorical",
    save_to_dir=base_dir+'/augmented/',
    save_prefix='aug',
    save_format='png'
)

# %% [markdown]
# ## Creación del Modelo

# %%
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers

# Creación del modelo
model = models.Sequential()

# Primera capa convolucional con ReLU y max pooling
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Segunda capa convolucional con ReLU y max pooling
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

# Tercera capa convolucional con ReLU y max pooling
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

# Cuarta capa convolucional con ReLU y max pooling
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

# Capa de aplanamiento
model.add(layers.Flatten())

# Capa densa con ReLU
model.add(layers.Dense(512, activation="relu"))

# Capa de salida con softmax para clasificación multiclase
model.add(layers.Dense(16, activation="softmax"))  # 16 clases

# Compilar el modelo
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=["acc"],
)

# %%
# Resumen del modelo
model.summary()

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Número de pasos por época
    epochs=30,  # Número de épocas
    validation_data=test_generator,
    validation_steps=50  # Número de pasos de validación
)

# Visualizar el proceso de entrenamiento
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Guardar el modelo
model.save('lego_brick_classifier.keras')


