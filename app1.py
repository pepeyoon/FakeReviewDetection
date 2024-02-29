from fastapi import FastAPI
import joblib
import string
import numpy as np
from pydantic import BaseModel, PositiveFloat  
from sklearn.feature_extraction.text import TfidfVectorizer


class ReviewRequest(BaseModel):
    text: str


ML_MODEL = joblib.load('mnb_model.joblib')
VECTORIZER = joblib.load('vectorizer.joblib')


api_title = 'fakereviewdetector'
api_description = 'A simple API to detect fake reviews'

api = FastAPI(title=api_title, description=api_description)


@api.post('/predict')
def predict_review(review_request: ReviewRequest):
    text = review_request.text
    # Split the text into words
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    # Join the words into a single string
    processed_text = " ".join(words)
    vectorized_text = VECTORIZER.transform([processed_text])
    prediction = ML_MODEL.predict(vectorized_text)[0]
    return {'prediction': str(prediction)}
#http://localhost:8000/docs