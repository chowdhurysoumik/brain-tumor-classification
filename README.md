# 🧠 Brain Tumor MRI Classification using Convolutional Neural Networks (CNN)

An end-to-end Deep Learning project for automated Brain Tumor Classification from MRI scans using Convolutional Neural Networks (CNN). This system classifies MRI images into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor. The project includes data preprocessing, augmentation, CNN model development, evaluation, visualization, model persistence, and an interactive Gradio-based web application for real-time predictions.

---

## 📌 Project Overview

Brain tumors are among the most critical neurological disorders, and early diagnosis plays a crucial role in treatment planning. Manual interpretation of MRI scans can be time-consuming and subject to human error. This project explores the application of Deep Learning techniques to automate the classification of brain MRI images.

The objective is to build a robust image classification pipeline capable of distinguishing between multiple tumor types and healthy MRI scans using a Convolutional Neural Network.

---

# 🎯 Problem Statement

Given an MRI scan of the brain, predict whether the image belongs to one of the following categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

This is a Multi-Class Image Classification problem.

---

# 🗂 Dataset

The project utilizes a Brain Tumor MRI Dataset containing MRI images organized into training and testing directories.

## Classes

| Class | Description |
|---------|-------------|
| Glioma | Tumor originating from glial cells |
| Meningioma | Tumor arising from the meninges |
| Pituitary | Tumor affecting the pituitary gland |
| No Tumor | Healthy brain MRI |

### Dataset Distribution

Training Images: 5712

Testing Images: 1311

Total Classes: 4

---

# ⚙️ Project Pipeline

The complete workflow consists of the following stages:

```text
MRI Dataset
     │
     ▼
Data Extraction
     │
     ▼
Image Preprocessing
     │
     ▼
Data Augmentation
     │
     ▼
CNN Architecture
     │
     ▼
Model Training
     │
     ▼
Model Evaluation
     │
     ▼
Visualization
     │
     ▼
Model Saving
     │
     ▼
Gradio Deployment
```

---

# 🔄 Data Preprocessing

The preprocessing stage ensures that the MRI images are suitable for Deep Learning training.

### Image Resizing

All MRI scans are resized to:

```python
224 × 224 × 3
```

to maintain a consistent input size across the dataset.

### Pixel Normalization

Pixel values are scaled from:

```python
0 - 255
```

to

```python
0 - 1
```

using:

```python
rescale = 1./255
```

This improves training stability and convergence speed.

---

# 🔀 Data Augmentation

To improve model generalization and reduce overfitting, image augmentation techniques are applied during training.

### Techniques Used

- Random Rotation
- Random Zoom
- Horizontal Flip

Implementation:

```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)
```

---

# 🧠 CNN Architecture

The model is built using TensorFlow and Keras Sequential API.

### Architecture

```text
Input Image (224×224×3)
          │
          ▼
Conv2D (32 Filters)
          │
          ▼
MaxPooling2D
          │
          ▼
Conv2D (64 Filters)
          │
          ▼
MaxPooling2D
          │
          ▼
Flatten
          │
          ▼
Dense (128 Neurons)
          │
          ▼
Dense (4 Neurons - Softmax)
```

### Layer Functions

#### Convolution Layer

Extracts visual features such as:

- Edges
- Shapes
- Textures
- Tumor patterns

#### Max Pooling Layer

Reduces feature map dimensions and computational complexity.

#### Flatten Layer

Converts 2D feature maps into a 1D feature vector.

#### Dense Layers

Perform classification based on learned features.

#### Softmax Layer

Produces probability scores for each tumor class.

---

# ⚡ Model Compilation

The model is compiled using:

### Optimizer

```python
Adam
```

Adaptive learning rate optimization algorithm.

### Loss Function

```python
Categorical Crossentropy
```

Used because the problem involves multiple classes.

### Evaluation Metric

```python
Accuracy
```

Measures classification performance.

---

# 🏋️ Model Training

The CNN model is trained using:

```python
model.fit()
```

with:

- Training Dataset
- Validation Dataset
- Multiple Epochs

During training, the model learns to identify tumor-specific patterns from MRI scans through backpropagation and weight optimization.

---

# 📊 Model Evaluation

The trained model is evaluated on unseen testing data.

### Evaluation Metrics

- Accuracy
- Loss
- Precision
- Recall
- F1 Score

Implementation:

```python
model.evaluate(test_data)
```

---

# 📈 Performance Visualization

Training and validation performance are visualized using:

### Accuracy Curve

Shows learning progression across epochs.

### Loss Curve

Illustrates optimization behavior and convergence.

These graphs help identify:

- Underfitting
- Overfitting
- Model stability

---

# 📋 Classification Report

A detailed classification report is generated containing:

- Precision
- Recall
- F1 Score
- Support

for each tumor category.

Example:

```python
classification_report()
```

---

# 🔍 Confusion Matrix

A confusion matrix is generated to visualize classification performance across all classes.

This helps identify:

- Correct classifications
- Misclassifications
- Class-specific weaknesses

Visualization is performed using:

```python
Seaborn Heatmap
```

---

# 💾 Model Persistence

The trained model is saved using:

```python
model.save()
```

allowing future inference without retraining.

Saved Format:

```text
brain_tumor_cnn.keras
```

---

# 🌐 Interactive Web Application

A Gradio-based web interface is developed for real-time MRI classification.

### Features

- Upload MRI image
- Automatic preprocessing
- Tumor prediction
- Confidence scores
- User-friendly interface

Workflow:

```text
Upload MRI
     │
     ▼
Preprocessing
     │
     ▼
CNN Prediction
     │
     ▼
Class Probabilities
     │
     ▼
Final Tumor Type
```

---

# 🧪 Prediction Example

Input:

MRI Brain Scan

Output:

```text
Predicted Class: Meningioma

Confidence Scores:

Meningioma : 92.8%
Glioma     : 7.2%
No Tumor   : 0.0%
Pituitary  : 0.0%
```

---

# 🛠 Technologies Used

### Deep Learning

- TensorFlow
- Keras

### Data Processing

- NumPy

### Visualization

- Matplotlib
- Seaborn

### Evaluation

- Scikit-Learn

### Deployment

- Gradio

### Environment

- Google Colab

---

# 📁 Project Structure

```text
BrainTumorProject/

│
├── dataset/
│   ├── Training/
│   └── Testing/
│
├── notebooks/
│   └── brain_tumor.ipynb
│
├── models/
│   └── brain_tumor_cnn.keras
│
├── app/
│   └── gradio_app.py
│
├── screenshots/
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   └── prediction_demo.png
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

# 🚀 Future Improvements

- Transfer Learning using VGG16
- Transfer Learning using ResNet50
- EfficientNet Implementation
- Grad-CAM Explainable AI Visualizations
- Model Comparison Dashboard
- Cloud Deployment
- Docker Containerization
- REST API Integration

---

# 🎓 Key Learning Outcomes

Through this project, the following concepts were explored:

- Image Preprocessing
- Data Augmentation
- Convolutional Neural Networks
- Multi-Class Classification
- Model Evaluation
- Deep Learning Workflow
- Medical Image Analysis
- Interactive Model Deployment

---

# ⭐ Results

The developed CNN model successfully learns discriminative MRI features and achieves strong classification performance on unseen data while providing an intuitive web-based prediction interface for end users.

This project demonstrates the complete lifecycle of a Deep Learning application, from raw medical imaging data to an interactive deployment-ready solution.