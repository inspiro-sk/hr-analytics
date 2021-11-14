FROM python:3.9.7

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY knn_model.joblib /app/knn_model.joblib

ENV FLASK_APP serve.py

CMD ["flask", "run", "--host=0.0.0.0"]