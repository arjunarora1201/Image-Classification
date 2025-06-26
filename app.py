import os
import numpy as np
import cv2
import uuid
from flask import Flask, request, render_template, jsonify
from skimage.feature import hog
from skimage.color import rgb2gray
from joblib import load
from scipy import ndimage as ndi
from skimage.feature import local_binary_pattern

app = Flask(__name__)
UPLOAD_FOLDER = "C:/Projects/Object Recognition/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

print("Loading models and scalers...")

scaler_hog = load("C:/Projects/Object Recognition/scalers/hog_scaler.pkl")
scaler_color = load("C:/Projects/Object Recognition/scalers/color_scaler.pkl")
scaler_texture = load("C:/Projects/Object Recognition/scalers/texture_scaler.pkl")
scaler_gabor = load("C:/Projects/Object Recognition/scalers/gabor_scaler.pkl")
pca = load("C:/Projects/Object Recognition/models/pca_model(45k).pkl")
model = load("C:\Projects\Object Recognition\models\svm_best_model(45k)_3fit_BestParam.pkl")

print("Models and scalers loaded successfully")

hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2)
}

def extract_multiscale_hog(image):
    gray = rgb2gray(image) if len(image.shape) == 3 else image
    hog1 = hog(gray, orientations=hog_params['orientations'], pixels_per_cell=hog_params['pixels_per_cell'], cells_per_block=hog_params['cells_per_block'], visualize=False)
    resized = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
    downsampled_pixels = (max(1, hog_params['pixels_per_cell'][0] // 2), max(1, hog_params['pixels_per_cell'][1] // 2))
    hog2 = hog(resized, orientations=hog_params['orientations'], pixels_per_cell=downsampled_pixels, cells_per_block=hog_params['cells_per_block'], visualize=False)
    return np.hstack([hog1, hog2])

def extract_texture_features(image,P=8,R=1):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_uint8 = (gray * 255).astype('uint8')
    lbp = local_binary_pattern(gray_uint8, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=P+2, range=(0, P+2), density=True)
    return hist

def extract_gabor_features(img):
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    sigmas = [1, 3]
    frequencies = [0.05, 0.25]
    
    gray = rgb2gray(img)
    gabor_features = []

    for theta in thetas:
        for sigma in sigmas:
            for frequency in frequencies:
                x, y = np.meshgrid(np.arange(gray.shape[1]), np.arange(gray.shape[0]))
                rotx = x * np.cos(theta) + y * np.sin(theta)
                gabor_real = ndi.gaussian_filter(gray, sigma) * np.cos(2*np.pi*frequency*rotx)

                mean_response = np.mean(gabor_real)
                var_response = np.var(gabor_real)
                energy = np.sum(gabor_real**2)

                gabor_features.extend([mean_response, var_response, energy])

    return np.array(gabor_features)

def extract_color_histogram(image, bins=32):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)

    feature_vector = np.hstack([hist_h, hist_s, hist_v]).flatten()
        
    return feature_vector


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hog_features = extract_multiscale_hog(image)
    color_features = extract_color_histogram(image)
    texture_features = extract_texture_features(image)
    gabor_features = extract_gabor_features(image)

    hog_features = scaler_hog.transform([hog_features])
    color_features = scaler_color.transform([color_features])
    texture_features = scaler_texture.transform([texture_features])
    gabor_features = scaler_gabor.transform([gabor_features])

    combined_features = np.hstack([hog_features, color_features, texture_features, gabor_features])
    combined_features = combined_features.reshape(1, -1) 

    transformed_features = pca.transform(combined_features)

    return transformed_features

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    print("Received prediction request")
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")
    file.save(file_path)
    
    try:
        features = preprocess_image(file_path)
        prediction = model.predict(features)[0]
        return jsonify({'prediction': int(prediction)})
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500
    
    finally:
        os.remove(file_path)
    
if __name__ == '__main__':
    app.run(debug=True)