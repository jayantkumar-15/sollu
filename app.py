from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
#model = .load('C:/Users/91814/Desktop/diabetes-prediction/model/RandomForest_model.pkl')
import pickle
from joblib import load

# Load the model from the specified path
model = load('notebooks/randomforestcl.pkl')

# Load the model from the .pkl file
#with open('C:/Users/91814/Desktop/diabetes-prediction/notebooks/randomforestcl.pkl', 'rb') as file:
    #model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the form
    gender = request.form['gender']
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    bmi = float(request.form['bmi'])
    smoking_history = request.form['smoking_history']
    HbA1c_level = int(request.form['HbA1c_level'])
    blood_glucose_level = int(request.form['blood_glucose_level'])
    diabetes = int(request.form['diabetes'])

    # Convert categorical variables to numerical
    gender = 1 if gender.lower() == 'male' else 0
    smoking_history = 1 if smoking_history.lower() == 'yes' else 0

    # Create an array of the input features
    features = np.array([[gender, age, hypertension, heart_disease, bmi, smoking_history, HbA1c_level, blood_glucose_level, diabetes]])

    # Make prediction
    prediction = model.predict(features)
    prediction_text = "Positive" if prediction[0] == 1 else "Negative"

    return render_template('result1.html', prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)

app.run(host='0.0.0.0', port=8080)


    

