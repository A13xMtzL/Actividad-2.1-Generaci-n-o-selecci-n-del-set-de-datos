# LEGO Brick Images Dataset
## TC3002B
### Alejandro Martínez Luna - A01276785

## Tabla de Contenido

- [Descripción](#descripción)
- [Fuente de Datos](#fuente-de-datos)
- [Información Detallada](#información-detallada)
- [Conjunto de Datos](#conjunto-de-datos)
- [Implementación del Modelo](#implementación-del-modelo)
    - [Modelo](#modelo)
        - [Resumen del Modelo](#resumen-del-modelo)
        - [Entrenamiento](#entrenamiento)
- [Ejecución del Código](#ejecución-del-código)

## Descripción

El dataset empleado contiene **40,000 imágenes** de **50 ladrillos LEGO diferentes**. Las imágenes fueron recopiladas y etiquetadas para su uso en tareas de clasificación y análisis de imágenes relacionadas con los ladrillos LEGO.

## Fuente de Datos

Las imágenes fueron obtenidas del conjunto de datos titulado [Imágenes de Ladrillos LEGO en Kaggle](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images). Este conjunto de datos proporciona una amplia variedad de imágenes de ladrillos LEGO.

## Información Detallada

- **Fecha de Creación o Última Actualización:** El conjunto de datos fue creado originalmente por **Joost Hazelzet** y ha estado disponible en Kaggle durante varios años.
- **Autores o Creadores:** Joost Hazelzet.
- **Formato de los Datos:** Imágenes digitales (JPEG, PNG, etc.).
- **Tamaño del Conjunto de Datos:** 40,000 imágenes en total.
- **Distribución de Clases:**
    - Cada imagen representa un **ladrillo LEGO específico**.
    - Hay **50 clases diferentes** de ladrillos LEGO en el conjunto de datos.
- **Preprocesamiento Aplicado:** Las imágenes se proporcionan en su formato original, sin un preprocesamiento específico. Sin embargo, los usuarios pueden aplicar sus propias técnicas de preprocesamiento según sea necesario.

## Conjunto de Datos

El conjunto de datos original utilizado en este código no está incluido en este repositorio debido a su tamaño. Sin embargo, puedes acceder al conjunto de datos modificado desde este [enlace de Google Drive](https://drive.google.com/drive/folders/1Ue-ZbK7UUYzEtVTQOHjzfBG0p6RI8nik?usp=sharing).

El conjunto de datos está dividido en dos carpetas: `train` y `test`. 
La carpeta `train` contiene **4,461** imágenes, mientras que la carpeta `test` contiene **1,918** imágenes. Cada imagen está etiquetada con la clase correspondiente de ladrillo LEGO.

Las imágenes del conjunto de datos fueron obtenidas del [sitio de Kaggle](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images).

# Implementación del Modelo
Para la implementación del modelo, se realizó una investigación y revisión de los siguientes papers:

- ["The Current State of the Art in Multi-Label Image Classification Applied on LEGO Bricks"](https://repository.tudelft.nl/islandora/object/uuid%3Af594616d-d8c9-47af-bcba-46fc23699fd0)
- ["Learning Stackable And Skippable Lego Bricks For Efficient, Reconfigurable, And Variable-Resolution Diffusion Modeling"](https://arxiv.org/pdf/2310.06389.pdf)
- ["Photos and rendered images of LEGO bricks"](https://www.nature.com/articles/s41597-023-02682-2.pdf)

## Modelo

El modelo utilizado en este es InceptionV3, una red neuronal convolucional (CNN) pre-entrenada. Las CNN son particularmente efectivas para tareas de clasificación de imágenes debido a su capacidad para aprender características jerárquicas de las imágenes. 

La arquitectura del modelo consiste en múltiples capas convolucionales seguidas de capas de agrupación máxima para extraer características de las imágenes de entrada. Luego incluye capas completamente conectadas para la clasificación.

## Resumen del Modelo:

- **Forma de Entrada:** (150, 150, 3) - Representa las dimensiones de las imágenes de entrada (150x150 píxeles con 3 canales de color).
- **Capas Convolucionales:** InceptionV3 contiene múltiples capas convolucionales que se utilizan para extraer características de las imágenes de entrada.
- **Capas de Agrupación:** InceptionV3 utiliza capas de agrupación máxima para reducir las dimensiones espaciales de los mapas de características.
- **Capas Densas:** Una capa completamente conectada con 1024 neuronas y activación ReLU, seguida de la capa de salida con 16 neuronas (una para cada clase de bloque de LEGO) y activación softmax.
- **Función de Pérdida:** Entropía Cruzada Categórica - Adecuada para tareas de clasificación multiclase.
- **Optimizador:** SGD con una tasa de aprendizaje de 0.0001 y momentum de 0.9 - Un algoritmo de optimización que ajusta la tasa de aprendizaje durante el entrenamiento.
- **Métricas:** Precisión - Evalúa el rendimiento del modelo durante el entrenamiento y la validación.

## Entrenamiento:

El modelo se entrena en dos etapas:
  - **Etapa 1:**
    - Se utiliza un modelo base pre-entrenado (InceptionV3) para extraer características de las imágenes.
    - Se agregan capas personalizadas para la clasificación de los ladrillos LEGO.
    - Las capas del modelo base se congelan para evitar que se actualicen durante el entrenamiento.
    - Se compila el modelo con un optimizador RMSprop y se entrena durante 5 épocas.
  - **Etapa 2:**
    - Se descongelan algunas capas del modelo base para permitir el ajuste fino.
    - Se compila el modelo nuevamente con un optimizador SGD y se entrena durante 10 épocas adicionales.
- Se utilizan callbacks para el registro de eventos en TensorBoard y para guardar los mejores modelos durante el entrenamiento.
- El modelo se evalúa en un conjunto de datos de prueba después del entrenamiento.
- Finalmente, se guardan los mejores modelos y se visualiza el proceso de entrenamiento mediante gráficos de precisión y pérdida.

## Ejecución del Código

Puedes ejecutar el código en el cuaderno de Jupyter utilizando cualquier entorno de Python que admita cuadernos de Jupyter. Si deseas ejecutar el código en Google Colab, puedes descomentar las dos líneas al principio del cuaderno que montan tu Google Drive en el entorno de Colab y cambiar el directorio al camino de tu unidad. Asegúrate de cambiar la ruta al lugar donde hayas almacenado el conjunto de datos.
