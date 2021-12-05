from flask import Flask, request
import joblib
import numpy as np

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def index():
    return {'result': 'Prediction service running'}


@app.route('/predict', methods=['POST'])
def predict():
    pipe = joblib.load('knn_model.joblib')

    data_arr = list(request.json.values())
    X = np.array(data_arr).reshape(1, -1)
    score = pipe.predict(X)

    return {'result': str(score)}


if __name__ == 'main':
    app.run()
