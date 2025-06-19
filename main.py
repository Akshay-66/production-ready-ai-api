from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from prometheus_client import Histogram, Counter, start_http_server
import time

# To start prometheus at 8001 port
start_http_server(8001)

app = FastAPI()
classifier = pipeline("sentiment-analysis")

# Load the trained model and tokenizer from local directory
model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
trained_model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create pipelines
naive_classifier = pipeline("sentiment-analysis", device=-1)
trained_classifier = pipeline("sentiment-analysis", model=trained_model, tokenizer=tokenizer, device=-1)

# Metrics
PREDICTION_TIME = Histogram("prediction_duration_seconds", "Time_spent_processing_prediction")
REQUEST = Counter("prediction_request_total", "Total_requests")
SENTIMENT_SCORE = Histogram("sentiment_score", "histogram_of_sentiment_scores", buckets=[0, 0.25, 0.5, 0.75, 1.0])

class TextInput (BaseModel):
    text: str

class SentimentOutput (BaseModel):
    text: str
    sentiment: str
    score: float

# Naive Model
@app.post("/predict/naive", response_model=SentimentOutput)
async def predict_naive_sentiment(input_data: TextInput):
    REQUEST.inc()
    start_time = time.time()

    result = naive_classifier (input_data.text)[0]

    SENTIMENT_SCORE.observe(result["score"])
    PREDICTION_TIME.observe(time.time() - start_time)
    
    return SentimentOutput (
        text = input_data.text,
        sentiment = result["label"],
        score = result["score"]
    )

# Trained Model
@app.post("/predict/trained", response_model=SentimentOutput)
async def predict_trained_sentiment(input_data: TextInput):
    REQUEST.inc()
    start_time = time.time()

    result = trained_classifier (input_data.text)[0]

    SENTIMENT_SCORE.observe(result["score"])
    PREDICTION_TIME.observe(time.time() - start_time)

    return SentimentOutput (
        text = input_data.text,
        sentiment = result["label"],
        score = result["score"]
    )