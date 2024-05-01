import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model("M2_IA/Act_2.1/__model_best_new_2.keras")

# Lista de nombres de clases
class_names = [
    "11214 Bush 3M friction with Cross axle",
    "18651 Cross Axle 2M with Snap friction",
    "2357 Brick corner 1x2x2",
    "3003 Brick 2x2",
    "3004 Brick 1x2",
    "3005 Brick 1x1",
    "3022 Plate 2x2",
    "3023 Plate 1x2",
    "3024 Plate 1x1",
    "3040 Roof Tile 1x2x45deg",
    "3069 Flat Tile 1x2",
    "32123 half Bush",
    "3673 Peg 2M",
    "3713 Bush for Cross Axle",
    "3794 Plate 1X2 with 1 Knob",
    "6632 Technic Lever 3M",
] 

def add_background(image):
    # Convert the image to an array
    image_array = np.array(image)

    # Check if the image has an alpha channel
    if image_array.shape[2] == 4:
        # Split the image into RGB and alpha channels
        rgb = image_array[:, :, :3]
        alpha = image_array[:, :, 3]

        # Create a new image with the background color and the same size as the original image
        background = np.full_like(rgb, [255, 255, 255])

        # Blend the original image with the background
        blended = (1 - alpha / 255) * background + (alpha / 255) * rgb

        return blended.astype(np.uint8)
    else:
        return image_array
        

# Función para realizar la predicción
def predict_image():
    # Abrir el cuadro de diálogo para seleccionar la imagen
    file_path = filedialog.askopenfilename()

    # Cargar la imagen y redimensionarla a las dimensiones requeridas por el modelo
    img = Image.open(file_path).convert("RGB")
    img = img.resize((224, 224))

    # Agregar un fondo blanco a la imagen si tiene un canal alfa
    img_background = add_background(img)
    img_background = Image.fromarray(img_background)

    # Mostrar la imagen en la interfaz gráfica
    img_preview = ImageTk.PhotoImage(img)
    img_label.config(image=img_preview)
    img_label.image = img_preview

    # Convertir la imagen a un arreglo numpy y normalizar los valores de píxeles
    img_array = np.array(img) / 255.0

    # Realizar la predicción utilizando el modelo
    prediction = model.predict(np.expand_dims(img_array, axis=0))

    # Obtener la clase predicha
    predicted_class = class_names[np.argmax(prediction)]

    # Actualizar la etiqueta con la clase predicha
    prediction_label.config(text=f"Predicted Class: {predicted_class}")


# Crear la ventana de la interfaz gráfica
root = tk.Tk()
root.title("LEGO Brick Classifier")

# Estilo para el botón de carga de imagen
button_style = {"background": "#4CAF50", "foreground": "white", "font": ("Arial", 12)}

# Botón para cargar la imagen y realizar la predicción
upload_button = tk.Button(
    root, text="Upload Image", command=predict_image, **button_style
)
upload_button.pack(pady=20)

# Etiqueta para mostrar la imagen
img_label = tk.Label(root)
img_label.pack()

# Estilo para la etiqueta de predicción
label_style = {"font": ("Arial", 14), "fg": "#333"}

# Etiqueta para mostrar la clase predicha
prediction_label = tk.Label(root, text="", **label_style)
prediction_label.pack()

# Ejecutar el bucle principal de la interfaz gráfica
root.mainloop()