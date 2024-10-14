# Use the official Python image from the Docker Hub
FROM python:3.9-slim

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
# Copy the model file explicitly
COPY logreg.pkl .

CMD ["python","flask_api.py"]


