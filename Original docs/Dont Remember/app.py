import os
import sys
import tempfile
import pickle
import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class MyotubeAnalyzer:
    def __init__(self):
        self.models = None
        self.feature_names = []
        self.metrics = {}

    def extract_features(self, image_path):
        """Extract features from a myotube image"""
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        features = {}

        # Basic image statistics
        features['mean_intensity'] = np.mean(gray)
        features['std_intensity'] = np.std(gray)
        features['min_intensity'] = np.min(gray)
        features['max_intensity'] = np.max(gray)

        # Color channel statistics
        for i, channel in enumerate(['blue', 'green', 'red']):
            features[f'{channel}_mean'] = np.mean(img[:, :, i])
            features[f'{channel}_std'] = np.std(img[:, :, i])

        # HSV statistics
        for i, channel in enumerate(['hue', 'saturation', 'value']):
            features[f'{channel}_mean'] = np.mean(hsv[:, :, i])
            features[f'{channel}_std'] = np.std(hsv[:, :, i])

        # Texture features using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['laplacian_var'] = laplacian.var()

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['num_contours'] = len(contours)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            features['mean_contour_area'] = np.mean(areas)
            features['max_contour_area'] = np.max(areas)
            features['total_contour_area'] = np.sum(areas)
        else:
            features['mean_contour_area'] = 0
            features['max_contour_area'] = 0
            features['total_contour_area'] = 0

        # Morphological features
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        features['opening_mean'] = np.mean(opening)
        features['closing_mean'] = np.mean(closing)

        return features

    def predict(self, image_path):
        """Predict myotube metrics for a new image"""
        if not self.models:
            raise ValueError("Models not trained yet")

        features = self.extract_features(image_path)
        if features is None:
            return None

        # Add passage and well (set to 0 for new images)
        features['passage'] = 0
        features['well'] = 0

        # Convert to DataFrame with correct column order
        features_df = pd.DataFrame([features])[self.feature_names]

        predictions = {}
        for target, model in self.models.items():
            predictions[target] = model.predict(features_df)[0]

        return predictions

    def load_models(self, filepath):
        """Load trained models from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.models = model_data['models']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']


# Initialize Flask app
app = Flask(__name__, static_folder='src/static')
CORS(app)

# Initialize analyzer
analyzer = MyotubeAnalyzer()
model_path = os.path.join(os.path.dirname(__file__), 'src', 'myotube_models.pkl')

try:
    analyzer.load_models(model_path)
    print(f"Models loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading models: {e}")
    analyzer = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if analyzer is None:
        return jsonify({'error': 'Models not loaded. Please check server configuration.'}), 500

    # Check if image file is in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, TIFF, or TIF files.'}), 400

    try:
        # Create temporary file to save uploaded image
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=os.path.splitext(secure_filename(file.filename))[1]) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        # Analyze the image
        predictions = analyzer.predict(temp_path)

        # Clean up temporary file
        os.unlink(temp_path)

        if predictions is None:
            return jsonify({'error': 'Failed to analyze image. Please check if the image is valid.'}), 400

        # Return predictions
        return jsonify({
            'success': True,
            'predictions': predictions,
            'message': 'Image analyzed successfully'
        })

    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass

        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the service is running"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': analyzer is not None,
        'message': 'Myotube Analyzer API is running'
    })


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)

