from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



# Paths for models
model_paths = {
    'ann': '/home/anuj/Documents/Deep learning/Project/laddu/models/ANN_Model/ann_model.pkl',
    'lru': '/home/anuj/Documents/Deep learning/Project/laddu/models/lru_model/Lru_model.pkl',
    'lstm': '/home/anuj/Documents/Deep learning/Project/laddu/models/lstm/Lstm_model.pkl'
}

# Globals to hold models
models = {}

# Function to load models
def load_models():
    global models
    try:
        # Load models
        for model_name, path in model_paths.items():
            with open(path, 'rb') as f:
                models[model_name] = pickle.load(f)

        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Load models when the app starts
load_models()

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index2.html')

def predict(x, model_name):
    """Predict using the selected model."""
    if x.size == 0:
        raise ValueError("Input data is empty.")



    # Retrieve the selected model
    selected_model = models.get(model_name)
    if not selected_model:
        raise ValueError(f"Model '{model_name}' not found.")

    # Make probability predictions
    if model_name in ['ann', 'lstm']:
        if hasattr(selected_model, 'predict_proba'):
            y_prob = selected_model.predict_proba(X_test)[:, 1]  # Probability of class 1
        else:
            y_prob = selected_model.predict(X_test).flatten()  # Adjust based on your model's API
    else:
        y_prob = selected_model.predict(X_test)  # Adjust as necessary

    # Apply thresholding if necessary
    if model_name in ['ann', 'lstm']:
        y_pred = np.where(y_prob > 0.5, 1, 0)
    else:
        y_pred = y_prob  # LRU and other models may return labels directly

    return y_pred, y_prob

@app.route('/uploadCsv', methods=['POST'])
def upload_file():
    """Handle CSV upload and predictions."""
    model_name = request.form.get('model', 'ann')

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({"error": "Please upload a valid CSV file."}), 400

    try:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)

        # Read the CSV file
        data = pd.read_csv(file_path)
        if data.empty:
            return jsonify({"error": "Uploaded CSV file is empty."}), 400

        # Make predictions
        predictions, probabilities = predict(data.values, model_name)

        # Prepare data for graph (e.g., histogram of probabilities)
        graph_data = {
            "labels": [f"Sample {i+1}" for i in range(len(probabilities))],
            "probabilities": probabilities.tolist()
        }

        return jsonify({
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "graph_data": graph_data
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/getPrediction', methods=['GET'])
def get_prediction():
    """Handle direct input predictions via query parameters."""
    try:
        queryParams = request.args.to_dict(flat=True)
        inputData = np.array([float(value) for value in queryParams.values()]).reshape(1, -1)
        model_name = request.args.get('model', 'ann')

        if inputData.size == 0:
            return render_template('results2.html', error="No input data provided.")

        # Make predictions
        predictions, probabilities = predict(inputData, model_name)

        # Prepare data for graph (single point)
        graph_data = {
            "labels": ["Input Data"],
            "probabilities": probabilities.tolist()
        }

        return render_template('results.html', 
                               predictions=predictions, 
                               probabilities=probabilities, 
                               graph_data=graph_data)
    except Exception as e:
        return render_template('results.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
