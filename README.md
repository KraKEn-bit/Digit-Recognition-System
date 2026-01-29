# Real-Time MNIST Handwritten Digit Recognition

An end-to-end digit recognition system featuring a **TensorFlow-trained backend** and a **custom NumPy-based inference engine**. This project includes a live drawing interface that bridges web technologies (HTML5/JS) with Python to perform real-time predictions inside **Google Colab**.

This project demonstrates both the full ML workflow (training, testing, evaluation) and the inner workings of a neural network (manual forward propagation with NumPy).

---

##  Project Overview

This system allows you to **draw digits on a canvas** and receive instant predictions. It also serves as a deep-dive into the mechanics of neural networks by demonstrating:

- **Training & Evaluation:** Building a model using TensorFlow/Keras on the MNIST dataset.  
- **Manual Math:** Performing inference using only NumPy to execute matrix multiplications and activation functions manually.  
- **Real-Time Bridge:** Using `google.colab.output` to facilitate communication between the browser's JavaScript and the Python kernel.

The MNIST dataset contains 28×28 grayscale images of handwritten digits (0–9).

---

## 🧠 Model Architecture

### TensorFlow Neural Network (Training)

| Layer           | Units | Activation |
|-----------------|-------|------------|
| Input Layer     | 784   | -          |
| Hidden Layer 1  | 128   | Sigmoid    |
| Hidden Layer 2  | 256   | Sigmoid    |
| Output Layer    | 10    | Softmax    |

The model is trained using **backpropagation** and **gradient descent**.

### NumPy Manual Inference (Real-Time Prediction)

| Layer           | Units | Activation |
|-----------------|-------|------------|
| Input Layer     | 784   | -          |
| Hidden Layer 1  | 128   | ReLU       |
| Hidden Layer 2  | 256   | ReLU       |
| Output Layer    | 10    | Softmax    |

The forward pass (inference) is calculated manually using NumPy, providing transparency into the neural network computations.

---

##  Features

- **Loads and preprocesses the MNIST dataset**  
- **TensorFlow Model:** Train and evaluate a neural network on MNIST  
- **Manual Inference:** Dot products, ReLU, and Softmax implemented in NumPy  
- **Interactive UI:** Draw digits directly in the notebook using a mouse  
- **Real-Time Predictions:** Predicts digits drawn on a web-based canvas  
- **Robot Vision Mode:** Visualizes the 28×28 grayscale input as seen by the model  
- **Prediction Display:** Shows predicted digit alongside actual drawn input  

---

## 📈 Results

- High classification accuracy on MNIST test data  
- Successfully predicts handwritten digits drawn on the canvas  
- Visual feedback helps debug and understand the model's perception  

---


##  Limitations & Challenges

While this project demonstrates real-time digit recognition, there are several limitations to note:

### 1. MLP (Multi-Layer Perceptron) Limitations
- **Spatial Blindness:** Flattening 28×28 images into a 1D vector (784 pixels) destroys the spatial structure of the digit.  
- **Translation Sensitivity:** Digits drawn off-center or in different positions are often misclassified.  
- **Scale Sensitivity:** Small, thin, or unusually shaped digits may not be recognized correctly.  
- **Random Digit Drawings:** If digits are drawn in unconventional styles or with irregular strokes, predictions become unreliable.  

### 2. CNN (Convolutional Neural Network) Considerations (Future Improvement)

### 3. Other Challenges
- **Manual NumPy Inference:** While educational, manually implemented forward propagation lacks optimizations like batching, which can slow down predictions.  
- **Interactive Canvas Sensitivity:** Drawing too quickly, with inconsistent stroke thickness, or in tiny sections of the canvas can result in incorrect predictions.  
- **Limited Dataset Variance:** The MNIST dataset contains mostly centered, standard-style digits. Hand-drawn digits that deviate from this style are harder to predict.  

> ⚠️ Overall: The current MLP system works well for standard, centered digits but struggles with random, off-center, or unusually styled digits. Upgrading to a CNN would address most of these issues.





---

## ⚙️ Setup & Installation

To run this project locally or in Google Colab:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/KraKEn-bit/Digit-Recognition-System.git
   cd Digit-Recognition-System
