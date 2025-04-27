import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model as keras_load_model
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import spacy

nlp = spacy.load("en_core_web_sm")
PICKLE_PATH = "data.pickle"
MODEL_PATH = "model.keras" 
INTENTS_PATH = "intents.json"

app = Flask(__name__)

model = None
words = []
labels = []
intents_data = None

chatbot = ChatBot(
    'EyeBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///chatterbot.sqlite3'
)

def load_intents():
    global intents_data
    with open(INTENTS_PATH, encoding='utf-8') as file:
        intents_data = json.load(file)

def load_assets():
    global model, words, labels

    if os.path.exists(PICKLE_PATH):
        with open(PICKLE_PATH, "rb") as f:
            words, labels, _, _ = pickle.load(f)
    else:
        raise FileNotFoundError("Pickle file not found!")

    if os.path.exists(MODEL_PATH):
        model = keras_load_model(MODEL_PATH)
    else:
        raise FileNotFoundError("Model file not found!")

def bag_of_words(sentence, words):
    bag = [0] * len(words)
    s_words = sentence.split()
    for word in s_words:
        if word in words:
            bag[words.index(word)] = 1
    return np.array(bag)

def predict_intent(sentence):
    bow = bag_of_words(sentence, words)
    prediction = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, prob] for i, prob in enumerate(prediction) if prob > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        return labels[results[0][0]], results[0][1]
    else:
        return None, 0.0

def get_response(tag):
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])
    return "I'm sorry, I didn't understand that."

@app.route("/chat", methods=["POST"])
def chat():
    if model is None or not words or not labels or intents_data is None:
        return jsonify({"error": "Model or data not loaded."}), 500

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "No message provided."}), 400

    user_message = data["message"]
    tag, confidence = predict_intent(user_message)

    if tag and confidence > 0.7:
        response = get_response(tag)
    else:
        response = str(chatbot.get_response(user_message))

    return jsonify({
        "response": response
    })


if __name__ == "__main__":
    load_intents()
    load_assets()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)