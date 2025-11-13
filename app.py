# app.py

# import the necessary libraries
from flask import Flask, request, render_template
import pickle
import numpy as np

# initialize the flask application
app = Flask(__name__)

# Load the pre-trained model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Home route
@app.route("/")
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get input values from form
        sepal_legnth = float(request.form['sepal_legnth'])
        sepal_width = float(request.form['sepal_width'])
        petal_legnth = float(request.form['petal_legnth'])
        petal_width = float(request.form['petal_width'])

        # prepare the feature array
        features = np.array([[sepal_legnth, sepal_width, petal_legnth, petal_width]])

        # make prediction
        prediction = model.predict(features)

        # since the model already returns class name, use it directly
        predicted_species = prediction[0]

        return render_template('index.html', prediction=f'The predicted species is {predicted_species}')

    except Exception as e:
        # in case of error, show a friendly message
        return render_template('index.html', prediction=f"Error: {str(e)}")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
