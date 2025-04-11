from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from models.depth_estimation import DepthEstimationModel
import yaml
import numpy as np
import cv2

app = Flask(__name__)

# Load model
with open('configs/model_params.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = DepthEstimationModel(config)
model.load_state_dict(torch.load('weights/depth_model.pth', map_location='cpu'))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read and preprocess image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict depth
        with torch.no_grad():
            depth = model(input_tensor)
            depth = model.scale_depth(depth)
        
        # Convert to colormap for visualization
        depth_np = depth.squeeze().cpu().numpy()
        depth_np = (depth_np * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
        
        # Save temporary result
        result_path = os.path.join('static', 'results', 'depth_result.jpg')
        cv2.imwrite(result_path, depth_colormap)
        
        return jsonify({
            'success': True,
            'result_url': f'/static/results/depth_result.jpg?{os.path.getmtime(result_path)}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    os.makedirs(os.path.join('static', 'results'), exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)