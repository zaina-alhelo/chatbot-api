# mainc.py

import nltk
import random
import json

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def load_data(file):
    with open(file, 'r') as f:
        return json.load(f)

def predict_intent(user_input, words, labels, model):
    #
    tag = "greeting"  
    confidence = 0.85  
    return tag, confidence

def get_responses(tag, intents_data):
    if tag in intents_data:
        return intents_data[tag]["responses"]
    return []


chatbot = None  
words = ["hello", "hi", "bye"]

labels = ["greeting", "farewell"]  
model = None  
