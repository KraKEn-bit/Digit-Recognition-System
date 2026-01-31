import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
import numpy as np

# CNN MODEL 
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# FLASK APP 
app = Flask(__name__, static_folder="static")

device = "cpu"
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pixels = request.json["pixels"]
    img = np.array(pixels).reshape(28, 28)
    img = img / 255.0
    img = 1 - img  # MNIST style

    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred = model(x).argmax(1).item()

    return jsonify({"prediction": pred})

if __name__ == "__main__":
    app.run(debug=True)
