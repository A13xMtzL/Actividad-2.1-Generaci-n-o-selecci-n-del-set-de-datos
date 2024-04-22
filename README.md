# LEGO Brick Images Dataset
### TC3002B
### Alejandro Martinez Luna - A01276785

## Description

This dataset contains **40,000 images** of **50 different LEGO bricks**. The images were collected and labeled for use in image classification and analysis tasks related to LEGO bricks.

## Data Source

The images were obtained from the dataset titled [Images of LEGO Bricks on Kaggle](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images). This dataset provides a wide variety of LEGO brick images, making it ideal for computer vision and machine learning projects.

## Detailed Information

- **Creation or Last Update Date:** The dataset was originally created by **Joost Hazelzet** and has been available on Kaggle for several years.
- **Authors or Creators:** Joost Hazelzet.
- **Data Format:** Digital images (JPEG, PNG, etc.).
- **Dataset Size:** 40,000 images in total.
- **Class Distribution:**
    - Each image represents a **specific LEGO brick**.
    - There are **50 different classes** of LEGO bricks in the dataset.
- **Applied Preprocessing:** The images are provided in their original format, without specific preprocessing. However, users can apply their own preprocessing techniques as needed.

## Dataset

The original dataset used in this code is not included in this repository due to its size. However, you can access the modified dataset from this [Google Drive link](https://drive.google.com/drive/folders/1Ue-ZbK7UUYzEtVTQOHjzfBG0p6RI8nik?usp=sharing). 

The dataset is divided into two folders: `train` and `test`. 
The `train` folder contains **4,461** images, while the `test` folder contains **1,918** images. Each image is labeled with the corresponding LEGO brick class.

The images of the dataset were obtained from the [Kaggle site](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images).

## Model

The machine learning model used in this code is a Convolutional Neural Network (CNN). CNNs are particularly effective for image classification tasks due to their ability to learn hierarchical features from images. The model architecture consists of multiple convolutional layers followed by max-pooling layers to extract features from the input images. It then includes fully connected layers for classification.

### Model Summary:

- **Input Shape:** (150, 150, 3) - Represents the dimensions of the input images (150x150 pixels with 3 color channels).
- **Convolutional Layers:** Four convolutional layers with increasing numbers of filters (32, 64, 128, 128) and ReLU activation functions.
- **Pooling Layers:** Four max-pooling layers to reduce the spatial dimensions of the feature maps.
- **Dense Layers:** One fully connected layer with 512 neurons and ReLU activation, followed by the output layer with 16 neurons (one for each LEGO brick class) and softmax activation.
- **Loss Function:** Categorical Crossentropy - Suitable for multi-class classification tasks.
- **Optimizer:** RMSprop - An adaptive learning rate optimization algorithm.
- **Metrics:** Accuracy - Evaluates the model's performance during training and validation.

### Training:

- **Number of Epochs:** 30 - Number of times the entire training dataset is passed forward and backward through the neural network.
- **Batch Size:** 25 - Number of training samples utilized in one iteration.
- **Steps per Epoch:** 100 - Total number of steps (batches of samples) before declaring one epoch finished.
- **Validation Steps:** 50 - Total number of steps (batches of samples) to draw before stopping when performing validation at the end of each epoch.

## Running the Code

You can run the code in the Jupyter notebook using any Python environment that supports Jupyter notebooks. If you want to run the code in Google Colab, you can uncomment the two lines at the beginning of the notebook that mount your Google Drive to the Colab environment and change the directory to the path of your drive. Make sure to change the path to the location where you have stored the dataset.

