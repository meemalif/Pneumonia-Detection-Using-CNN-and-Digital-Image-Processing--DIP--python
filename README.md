# Pneumonia Detection Using CNN and Digital Image Processing (DIP)

## Overview
This project leverages advanced imaging techniques and Convolutional Neural Networks (CNN) to detect pneumonia from chest X-ray images. The model is designed to enhance and analyze medical images, facilitating accurate diagnosis.

## Folder Structure
project-root/
│
├── data/
│ ├── train/
│ │ ├── NORMAL/
│ │ └── PNEUMONIA/
│ ├── test/
│ │ ├── NORMAL/
│ │ └── PNEUMONIA/
│ └── val/
│ ├── NORMAL/
│ └── PNEUMONIA/
│
├── notebooks/
│ └── try2.ipynb
│
└── models/
└── cnn_model.h5

bash
Copy code

## Running the Code
To run the code and train your own model, follow these steps:

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install Dependencies
Make sure you have Python and Jupyter Notebook installed. Install the required packages using:

```bash
pip install -r requirements.txt
```

3. Prepare the Dataset
Ensure your dataset is structured as described in the folder structure. You should have the images categorized into NORMAL and PNEUMONIA folders within train, test, and val directories.

4. Run the Jupyter Notebook
Open the Jupyter Notebook and run all cells to train the model:

```bash
jupyter notebook notebooks/try2.ipynb
```
Training Your Own Model
Dataset Preparation
Data Structure:

<strong> Training Data: </strong> data/train/NORMAL and data/train/PNEUMONIA <br>
 Testing Data: data/test/NORMAL and data/test/PNEUMONIA <br>
Validation Data: data/val/NORMAL and data/val/PNEUMONIA <br>
Preprocessing Steps:

Convert images to grayscale.
Resize images to 150x150 pixels.
Normalize pixel values.
Data Augmentation
To handle the data imbalance and improve model generalization, employ data augmentation techniques such as:

Rotation
Zooming
Flipping
Model Architecture
The CNN architecture comprises:

Convolutional Layers: For feature extraction.
Pooling Layers: For dimensionality reduction.
Dense Layers: For classification.
The model progressively abstracts and identifies pneumonia indicators from the X-ray images.

Training
The model uses the following parameters:

Optimizer: ADAMS and RMSprop
EarlyStopping: Monitor validation accuracy with patience of 2
ReduceLROnPlateau: Monitor validation accuracy, with a factor of 0.3 and minimum learning rate of 0.000001
Understanding the Project
This project aims to detect pneumonia using chest X-ray images. Pneumonia inflames the air sacs in the lungs, and early detection is crucial for effective treatment. The dataset consists of 5863 pediatric chest X-ray images, divided into 'Pneumonia' and 'Normal' categories. The images undergo preprocessing, including grayscale conversion, resizing, and normalization. Data augmentation is applied to address data imbalance.

The CNN model is designed with layers to extract features, reduce dimensions, and classify images. Training involves monitoring validation accuracy to prevent overfitting and optimizing learning rates.

The project's future work includes expanding the dataset, incorporating diverse age groups, exploring 3D imaging, and integrating more diseases like Covid for comprehensive analysis.

<h2>Acknowledgements </h2>
This project was developed by Muneeb Ahmad and Rayan Lakhani