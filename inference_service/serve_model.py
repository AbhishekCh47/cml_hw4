import os
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn

# Define the CNN model
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.c1 = nn.Conv2d(1, 32, 3, 1)
        self.c2 = nn.Conv2d(32, 64, 3, 1)
        self.d1 = nn.Dropout(0.25)
        self.d2 = nn.Dropout(0.5)
        self.fc_a = nn.Linear(9216, 128)
        self.fc_b = nn.Linear(128, 10)

    def forward(self, img):
        img = self.c1(img)
        img = F.relu(img)
        img = self.c2(img)
        img = F.relu(img)
        img = F.max_pool2d(img, 2)
        img = self.d1(img)
        img = torch.flatten(img, 1)
        img = self.fc_a(img)
        img = F.relu(img)
        img = self.d2(img)
        img = self.fc_b(img)
        return F.log_softmax(img, dim=1)

# Initialize Flask app
webapp = Flask(__name__)

# Create upload folder
IMAGE_UPLOAD_DIR = 'user_uploads'
os.makedirs(IMAGE_UPLOAD_DIR, exist_ok=True)
webapp.config['UPLOAD_FOLDER'] = IMAGE_UPLOAD_DIR

# Load model on CPU
runtime_device = torch.device("cpu")
digit_model = DigitClassifier().to(runtime_device)
digit_model.load_state_dict(torch.load("/mnt/ac11950_model.pt", map_location=runtime_device))
digit_model.eval()

# Image transformation pipeline
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Home route
@webapp.route('/home')
def main_page():
    return render_template("mnist.html")

# Prediction route
@webapp.route('/classify-digit', methods=['POST'])
def classify_digit():
    if 'digitfile' not in request.files:
        return "No file uploaded", 400
    digitfile = request.files['digitfile']
    if digitfile.filename == '':
        return "Empty filename", 400

    img_obj = Image.open(digitfile).convert("L")
    input_tensor = preprocess(img_obj).unsqueeze(0)

    with torch.no_grad():
        result = digit_model(input_tensor)
        prediction = result.argmax(dim=1, keepdim=True).item()

    return render_template("mnist.html", prediction=prediction)

# Start app
if __name__ == '__main__':
    webapp.run(host="0.0.0.0", port=5000, debug=True)
