# Real-Time MNIST Handwritten Digit Recognition

A real-time handwritten digit recognition system using the **MNIST dataset**. This project demonstrates both a **TensorFlow-trained neural network** and a **custom NumPy-based MLP for manual inference**, complete with an interactive drawing interface running inside **Google Colab**.

---

## 🚀 Project Overview

This project allows users to **draw digits on a web-based canvas**, which are then processed and predicted in real-time by a neural network. The system demonstrates:

1. **Training & testing a neural network** on MNIST using TensorFlow.  
2. **Manual forward propagation** using NumPy for inference, showing the math behind predictions.  
3. **Interactive real-time predictions** via an HTML5 canvas inside Colab.

The MNIST dataset contains 28×28 grayscale images of handwritten digits (0–9).

---

## 🧠 Model Architectures

### TensorFlow Neural Network (Training)

| Layer           | Units | Activation |
|-----------------|-------|------------|
| Input Layer     | 784   | -          |
| Hidden Layer 1  | 128   | Sigmoid    |
| Hidden Layer 2  | 256   | Sigmoid    |
| Output Layer    | 10    | Softmax    |

### NumPy Manual Inference (Real-Time Prediction)

| Layer           | Units | Activation |
|-----------------|-------|------------|
| Input Layer     | 784   | -          |
| Hidden Layer 1  | 128   | ReLU       |
| Hidden Layer 2  | 256   | ReLU       |
| Output Layer    | 10    | Softmax    |

---

## 📊 Features

- **TensorFlow Model:** Train and evaluate a neural network on MNIST.  
- **Manual Inference:** Implements forward propagation using NumPy arrays (dot products, ReLU, Softmax).  
- **Interactive Drawing:** Draw digits directly in the notebook with a mouse.  
- **Real-Time Predictions:** Uses `google.colab.output` callbacks to send Base64 image data from JS to Python.  
- **Robot Vision Mode:** Visualizes the resized 28×28 input as seen by the model.  
- **Prediction Display:** Shows predicted digit alongside actual drawn input.  

---

## 📈 Results

- High classification accuracy on MNIST test data.  
- Successfully predicts handwritten digits drawn on the canvas.  
- Visual feedback helps debug and understand the model's perception.  

---



