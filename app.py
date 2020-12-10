from math import expm1
import joblib
import pandas as pd

from flask import Flask, jsonify, request
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('assets/airbnb_price_pred.h5')
transformer = joblib.load('assets/data_transformer.joblib')

@app.route("/", methods=["POST"])
def index():
    data = request.json
    print(data) 
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(transformer.transform(df))
    predicted_price = expm1(prediction.flatten()[0])
    return jsonify({"price":str(predicted_price)})

# curl -d '{"neighbourhood_group": "Brooklyn", "latitude": 40.64749, "longitude": -73.97237, "room_type": "Private room", "minimum_nights": 1, "number_of_reviews": 9, "calculated_host_listings_count": 6, "availability_365": 365}' -H "Content-Type: application/json" -X POST http://localhost:5000
