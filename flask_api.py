from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

# Load the model
try:
    with open("logreg.pkl", "rb") as pickle_in:
        model = pickle.load(pickle_in)
except FileNotFoundError:
    print("Model file not found. Ensure 'logreg.pkl' is present in the correct directory.")
    model = None

# Home route serving an HTML UI
@app.route('/')
def home():
    return render_template('index.html')

# Route for form submission (UI input) for the /predict
@app.route('/predict', methods=["POST"])
def predict_class():
    try:
        age = int(request.form['age'])
        new_user = int(request.form['new_user'])
        total_pages_visited = int(request.form['total_pages_visited'])
        
        # Make prediction
        prediction = model.predict([[age, new_user, total_pages_visited]])
        
        # Return the prediction result
        prediction_text = f"Model prediction is: {prediction[0]}"
        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return str(e), 500

# File upload route for the /predict_file endpoint
@app.route('/predict_file', methods=["POST"])
def prediction_test_file():
    try:
        df_test = pd.read_csv(request.files.get("file"))
        
        required_columns = ["age", "new_user", "total_pages_visited"]
        if not all(column in df_test.columns for column in required_columns):
            return "CSV does not have the required columns.", 400
        
        prediction = model.predict(df_test)
        prediction_list = str(list(prediction))
        return render_template('index.html', prediction_text=f"File prediction: {prediction_list}")
    
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    if model:
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Exiting: Model file missing or failed to load.")
