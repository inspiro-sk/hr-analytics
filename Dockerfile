FROM python:3.9.7

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl /app/model.pkl
COPY scaler.pkl /app/scaler.pkl
COPY poly.pkl /app/poly.pkl
COPY serve.py /app/serve.py

ENV FLASK_APP serve.py

CMD ["flask", "run", "--host=0.0.0.0"]