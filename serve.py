from flask import Flask, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def index():
    return {'result': 'Prediction service running'}


@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    polynoms = pickle.load(open('poly.pkl', 'rb'))

    data_arr = list(request.json.values())

    X_scaled = scaler.transform(np.array(data_arr).reshape(1, -1))
    X_poly = polynoms.transform(X_scaled)

    score = model.predict(X_poly)[0]

    return {'result': str(score)}


if __name__ == 'main':
    app.run()
