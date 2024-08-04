from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
import os
app = Flask(__name__)
Swagger(app)
dirname = os.path.dirname(__file__)
model_path = os.path.join(dirname, '../model/knn_irish_model.pkl')
# Load the pre-trained model
try:
    with open(model_path, "rb") as model_file:
        classifier = pickle.load(model_file)
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

@app.route('/')
def welcome():
    return "Welcome to the IRIS Prediction API!"

@app.route('/predict', methods=["GET"])
def predict_iris():
    """Predict the IRIS species
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: sepal_length
        in: query
        type: number
        required: true
      - name: sepal_width
        in: query
        type: number
        required: true
      - name: petal_length
        in: query
        type: number
        required: true
      - name: petal_width
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """
    sepal_length = request.args.get("sepal_length")
    sepal_width = request.args.get("sepal_width")
    petal_length = request.args.get("petal_length")
    petal_width = request.args.get("petal_width")
    
    if not all([sepal_length, sepal_width, petal_length, petal_width]):
        return "Error: Missing one or more required parameters", 400
    
    features = np.array([[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])
    prediction = classifier.predict(features)
    species = ['setosa', 'versicolor', 'virginica']
    predicted_species = species[prediction[0]]
    
    return f"The predicted species is {predicted_species}"

@app.route('/predict_file', methods=["POST"])
def predict_iris_file():
    """Predict the IRIS species from a CSV file
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
    """
    df_test = pd.read_csv(request.files.get("file"))
    predictions = classifier.predict(df_test)
    species = ['setosa', 'versicolor', 'virginica']
    predicted_species = [species[pred] for pred in predictions]
    
    return str(predicted_species)

if __name__ == '__main__':
    if classifier:
        app.run(host='0.0.0.0', port=8000)
    else:
        print("Model is not loaded. Please check the model file.")
