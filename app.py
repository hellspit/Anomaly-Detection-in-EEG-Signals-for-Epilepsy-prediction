from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Load the model
ann_model_path = 'models/ANN_Model/ann_model.pkl'
try:
    with open(ann_model_path, 'rb') as f:
        loaded_model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

@app.route('/')
def home():
    return render_template('index2.html')

def predict(x):
    # Check if input data is empty
    if x.size == 0:
        raise ValueError("Input data is empty.")
    
    X_test = scaler.fit_transform(x)
    y_log = loaded_model.predict(X_test)
    y_pred = np.where(y_pred > 99, 99, y_pred)
    return y_pred

@app.route('/uploadCsv', methods=['POST'])
def uploadFile():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({"error": "Please upload a CSV file."}), 400

    upload_folder = app.config.get('UPLOAD_FOLDER', './uploads')
    os.makedirs(upload_folder, exist_ok=True)

    try:
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"File save failed: {str(e)}"}), 500

    try:
        data = pd.read_csv(file_path)
        # Check if the DataFrame is empty
        if data.empty:
            return jsonify({"error": "Uploaded CSV file is empty."}), 400
        
        predictions = predict(data.values)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/getPrediction', methods=['GET'])
def getPrediction():
    try:
        queryParams = request.args.to_dict(flat=True)
        inputData = np.array([float(value) for value in queryParams.values()]).reshape(1, -1)
        
        # Check for empty inputData
        if inputData.size == 0:
            return render_template('results.html', error="No input data provided.")
        
        predictions = predict(inputData)
        return render_template('results.html', predictions=predictions)
    except Exception as e:
        return render_template('results.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)

