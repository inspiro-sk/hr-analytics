from flask import Flask, request
import joblib
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

    pipe = joblib.load('knn_model.joblib')

    data_arr = list(data.values())
    X = np.array(data_arr).reshape(1, -1)

    score = pipe.predict(X)

    return {'result': str(score)}


if __name__ == 'main':
    app.run()
