FROM python:3.10-slim

# RUN apt-get update && apt-get upgrade -y && apt-get clean

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY main.py .
COPY ./model ./model

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]