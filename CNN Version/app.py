import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Added for connection fix
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
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  

# Loading model
device = "cpu"
model = CNN()
try:
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

# ROUTES
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        pixels = request.json["pixels"]
        
        img = np.array(pixels).reshape(28, 28)
        img = img / 255.0 

        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(x)
            pred = output.argmax(1).item()

        return jsonify({"prediction": str(pred)})
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)