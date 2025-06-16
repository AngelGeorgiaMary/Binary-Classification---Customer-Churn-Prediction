import xgboost as xgb
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load Booster
booster = xgb.Booster()
booster.load_model("xgb_model.json")

# Load the scaler if used
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract and encode input
        tenure = float(request.form['tenure'])
        monthly = float(request.form['monthlycharges'])
        total = float(request.form['totalcharges'])
        contract = request.form['contract']
        internet = request.form['internetservice']
        security = request.form['onlinesecurity']
        tech = request.form['techsupport']
        senior = request.form['seniorcitizen']
        paperless = request.form['paperlessbilling']
        payment = request.form['paymentmethod']

        # Manual encoding
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        internet_map = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
        binary_map = {'No': 0, 'Yes': 1}
        payment_map = {'Bank transfer': 0, 'Credit card': 1, 'Electronic check': 2, 'Mailed check': 3}

        features = [
            tenure,
            monthly,
            total,
            contract_map.get(contract, 0),
            internet_map.get(internet, 0),
            binary_map.get(security, 0),
            binary_map.get(tech, 0),
            binary_map.get(senior, 0),
            binary_map.get(paperless, 0),
            payment_map.get(payment, 0)
        ]

        # Preprocess
        features_scaled = scaler.transform([features])

        # XGBoost prediction
        dmatrix = xgb.DMatrix(features_scaled)
        prediction = booster.predict(dmatrix)
        result = "Churn" if prediction[0] > 0.5 else "No Churn"

        return render_template('index.html', prediction=result)

    return render_template('index.html', prediction=None)
