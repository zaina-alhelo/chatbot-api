import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model as keras_load_model
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import nltk
nltk.data.path.append('./nltk_data')

PICKLE_PATH = "data.pickle"
MODEL_PATH = "model.keras"
INTENTS_PATH = "intents.json"

app = Flask(__name__)

model = None
words = []
labels = []
intents_data = None

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

chatbot = ChatBot(
    'EyeBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///chatterbot.sqlite3',
    tagger_language='english', 
    use_nltk_tokenizer=True,   
    preprocessors=[
        'chatterbot.preprocessors.clean_whitespace'
    ]
)

def load_intents():
    global intents_data
    try:
        with open(INTENTS_PATH, encoding='utf-8') as file:
            intents_data = json.load(file)
        print("Intents data loaded successfully")
    except FileNotFoundError:
        print(f"Error: {INTENTS_PATH} not found!")
        intents_data = {"intents": []}
    except json.JSONDecodeError:
        print(f"Error: {INTENTS_PATH} is not a valid JSON file!")
        intents_data = {"intents": []}

def load_assets():
    global model, words, labels
    
    try:
        if os.path.exists(PICKLE_PATH):
            with open(PICKLE_PATH, "rb") as f:
                words, labels, _, _ = pickle.load(f)
            print("Pickle data loaded successfully")
        else:
            print(f"Warning: {PICKLE_PATH} not found!")
            words, labels = [], []
            
        if os.path.exists(MODEL_PATH):
            model = keras_load_model(MODEL_PATH)
            print("Model loaded successfully")
        else:
            print(f"Warning: {MODEL_PATH} not found!")
            model = None
            
    except Exception as e:
        print(f"Error loading assets: {str(e)}")
        words, labels = [], []
        model = None

def bag_of_words(sentence, words):
    bag = [0] * len(words)
    s_words = sentence.split()
    for word in s_words:
        if word in words:
            bag[words.index(word)] = 1
    return np.array(bag)

def predict_intent(sentence):
    if not model or not words or not labels:
        return None, 0.0
        
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
    if not intents_data:
        return "I'm sorry, I couldn't process your request."
        
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])
    return "I'm sorry, I didn't understand that."

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Chatbot API is running"})

@app.route("/health", methods=["GET"])
def health():
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "words_loaded": len(words) > 0,
        "labels_loaded": len(labels) > 0,
        "intents_loaded": intents_data is not None and len(intents_data.get("intents", [])) > 0
    }
    return jsonify(status)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "No message provided."}), 400

        user_message = data["message"]
        
        if model is not None and words and labels and intents_data is not None:
            tag, confidence = predict_intent(user_message)
            
            if tag and confidence > 0.7:
                response = get_response(tag)
            else:
                response = str(chatbot.get_response(user_message))
        else:
            response = str(chatbot.get_response(user_message))

        return jsonify({
            "response": response
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
def initialize_app():
    download_nltk_data()
    load_intents()
    load_assets()
    print("Application initialized successfully")

initialize_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)