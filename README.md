# Real-Time MNIST Digit Recognition (CNN)

An **interactive web-based handwritten digit recognizer** powered by a **PyTorch Convolutional Neural Network (CNN)**.  
Draw digits (0–9) on a canvas and get **instant predictions in real time**.

This project is an **upgrade from an earlier MLP + NumPy manual inference version**, created mainly for learning.  
👉 **The main and final project is the CNN-based version.**

---

##  Overview

- Draw digits directly in the browser  
- Get real-time predictions using a **CNN trained on MNIST (28×28 grayscale images)**  
- Compare **MLP vs CNN** to understand why CNNs perform better on image data  

---

## 🧠 Models Used:

### 🔹 Previous Version: MLP (Learning Purpose)
- Fully connected network (NumPy / TensorFlow)
- Required image flattening → lost spatial information  
- Struggled with off-center or irregular drawings  

> ⚠️ Kept only for **comparison and understanding**, not the main project.

---

### 🔹 Main Project: CNN (PyTorch) ✅
- Convolution + pooling layers preserve spatial features  
- More robust to scale, position, and drawing style  
- Higher accuracy and smoother real-time performance  

# **All core logic and deployment focus on the CNN version.**

---

##  Features

- Interactive HTML5 drawing canvas  
- Real-time digit prediction  
- PyTorch CNN backend  
- Clean, responsive web UI  
- MLP version included only for educational comparison  

---

##  Demo

A **GIF demo** is included in the repository showing:
- Drawing a digit  
- Real-time prediction output  


![Demo](https://github.com/KraKEn-bit/Digit-Recognition-System/blob/main/CNN%20Version/OnepieceDigitrecog-ezgif.com-speed.gif))

