# Real-Time MNIST Digit Recognition 

An **interactive web-based digit recognition system** built with **PyTorch CNN**. Draw digits (0–9) on the canvas and get **real-time predictions**.  

This project is an upgrade from a previous **MLP + NumPy manual inference version** and now features a **Convolutional Neural Network (CNN)** for better accuracy, robustness, and interactive experience.

---

## 🚀 Project Overview

This system allows you to:

- **Draw digits** on a canvas in your browser  
- **Predict in real-time** using a CNN model trained on MNIST  
- **Explore the difference** between MLP and CNN approaches  
- **Learn neural network mechanics** (from MLP manual inference to CNN training)  

The MNIST dataset contains **28×28 grayscale images** of handwritten digits (0–9).  

---

## 🧠 Model Architectures

### 1️⃣ Previous MLP Version (NumPy + TensorFlow)

| Layer           | Units | Activation |
|-----------------|-------|------------|
| Input Layer     | 784   | -          |
| Hidden Layer 1  | 128   | ReLU       |
| Hidden Layer 2  | 256   | ReLU       |
| Output Layer    | 10    | Softmax    |

**Limitations of MLP:**

- Flattening images destroys spatial structure → less robust  
- Sensitive to off-center, scaled, or oddly drawn digits  
- Random drawings may not be recognized correctly  

> ⚠️ Works well for standard, centered digits but struggles with real-world handwriting.

---

### 2️⃣ Current CNN Version (PyTorch)

| Layer           | Output Shape        | Activation |
|-----------------|------------------|------------|
| Conv2D (1→32)   | 26×26×32          | ReLU       |
| Conv2D (32→64)  | 24×24×64          | ReLU       |
| MaxPool2D (2x2) | 12×12×64          | -          |
| Flatten         | 9216               | -          |
| Fully Connected | 128                | ReLU       |
| Fully Connected | 10                 | -          |

**Advantages of CNN:**

- Preserves spatial structure → more robust recognition  
- Better at recognizing off-center, scaled, or irregular digits  
- Higher accuracy than MLP  
- Integrates seamlessly with interactive canvas  

---

##  Features:

- **Interactive Canvas:** Draw digits directly in the browser  
- **Real-Time Prediction:** Predict digits instantly  
- **Web-Based UI:** Modern, responsive interface with animated footer  
- **CNN Backend:** PyTorch-based convolutional neural network  
- **Previous MLP Version:** Kept for comparison and learning purposes  

---

## 📈 Results

- **High accuracy** on MNIST test set  
- **Correctly recognizes** digits drawn in various positions and styles  
- **Interactive feedback** helps users understand how the model perceives their input  

---

## Demo:

Open your browser at `http://127.0.0.1:5000/` after running:

```bash
python app.py
