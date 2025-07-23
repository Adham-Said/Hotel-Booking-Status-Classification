from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from preprocessing import apply_encoding, apply_scaler, feature_engineering, parse_dates, numerical_features, categorical_features

app = Flask(__name__)

with open('bundle.pkl', 'rb') as f:
    saved = pickle.load(f)
    model = saved['model']
    onehot_encoder = saved['onehot_encoder']
    log_transformer = saved['log_transformer']
    robust_scaler = saved['robust_scaler']
    feature_names = saved['feature_names']


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict", methods = ['POST'])
def predict():
    features = [request.form.get(feat) for feat in feature_names]
    features_df = pd.DataFrame([features], columns = feature_names)
    features_engineered = feature_engineering(features_df)
    features_date = parse_dates(features_engineered, 'date of reservation')
    encoded_features = apply_encoding(features_date, onehot_encoder)
    scaled_features = apply_scaler(encoded_features, numerical_features)
    prediction = model.predict(scaled_features)[0]
    if prediction == 0:
        prediction = 'Not Cancelled'
    else:
        prediction = "Cancelled"
    return render_template("index.html", prediction_text = f"The Reservation is Most likely to be {prediction}")



if __name__ == "__main__":
    app.run(debug = True)
