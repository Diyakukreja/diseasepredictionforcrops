from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
import torch
torch.hub.set_dir('/tmp')

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights='IMAGENET1K_V1')

# Modify the fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 42)
)

# Load the model weights
model.load_state_dict(torch.load('crop_disease_resnet50.pth', map_location=device))
model = model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = [
    'American Bollworm on Cotton', 'Anthracnose on Cotton', 'Army worm', 
    'Bacterial Blight in Rice', 'Brownspot', 'Common_Rust', 'Cotton Aphid', 
    'Flag Smut', 'Gray_Leaf_Spot', 'Healthy Maize', 'Healthy Wheat', 'Healthy cotton', 
    'Leaf Curl', 'Leaf smut', 'Mosaic sugarcane', 'RedRot sugarcane', 'RedRust sugarcane', 
    'Rice Blast', 'Sugarcane Healthy', 'Tungro', 'Wheat Brown leaf Rust', 'Wheat Stem fly', 
    'Wheat aphid', 'Wheat black rust', 'Wheat leaf blight', 'Wheat mite', 'Wheat powdery mildew', 
    'Wheat scab', 'Wheat___Yellow_Rust', 'Wilt', 'Yellow Rust Sugarcane', 'bacterial_blight in Cotton', 
    'bollrot on Cotton', 'bollworm on Cotton', 'cotton mealy bug', 'cotton whitefly', 
    'maize ear rot', 'maize fall armyworm', 'maize stem borer', 'pink bollworm in cotton', 
    'red cotton bug', 'thirps on cotton'
]
@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            _, predicted_class = torch.max(output, 1)
            predicted_class_name = class_names[predicted_class.item()]

        return jsonify({'disease': predicted_class_name}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Ensure compatibility with Render
    app.run(host='0.0.0.0', port=port)
