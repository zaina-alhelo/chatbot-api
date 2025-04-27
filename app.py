from flask import Flask, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
nltk.data.path.append('./nltk_data')
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import numpy as np
from tensorflow import keras
import random
import json
import pickle
import os

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

INTENTS_PATH = "intents.json"
PICKLE_PATH = "data.pickle"
MODEL_PATH = "model.keras"

app = Flask(__name__)

model = None
words = []
labels = []
intents_data = {}

def load_intents():
    if not os.path.exists(INTENTS_PATH):
        raise FileNotFoundError(f"{INTENTS_PATH} not found.")
    with open(INTENTS_PATH, encoding='utf-8') as file:
        return json.load(file)

def preprocess_data(data):
    words, labels, docs_x, docs_y = [], [], [], []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = sorted(set(lemmatizer.lemmatize(w.lower()) for w in words if w != "?"))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0] * len(labels)

    for i, doc in enumerate(docs_x):
        bag = [1 if w in [lemmatizer.lemmatize(word.lower()) for word in doc] else 0 for w in words]
        output_row = out_empty.copy()
        output_row[labels.index(docs_y[i])] = 1
        training.append(bag)
        output.append(output_row)

    return np.array(training), np.array(output), words, labels

def load_model():
    global model, words, labels, intents_data

    intents_data = load_intents()

    if os.path.exists(PICKLE_PATH):
        with open(PICKLE_PATH, "rb") as f:
            words, labels, training, output = pickle.load(f)
    else:
        training, output, words, labels = preprocess_data(intents_data)
        with open(PICKLE_PATH, "wb") as f:
            pickle.dump((words, labels, training, output), f)

    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
    else:
        model = keras.models.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(len(training[0]),)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(output[0]), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(training, output, epochs=600, batch_size=16, verbose=1)
        model.save(MODEL_PATH)

# تجهيز Bag of Words
def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# التنبؤ بالـ intent
def predict_intent(sentence):
    bow = bag_of_words(sentence, words)
    results = model.predict(np.array([bow]))[0]
    results_index = np.argmax(results)
    tag = labels[results_index]
    confidence = results[results_index]
    return tag, confidence

# جلب الرد بناءً على الـ intent
def get_response(tag):
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I don't understand."

# صفحة البداية
@app.route("/", methods=["GET"])
def home():
    return "🤖 Bot is running!"

# نقطة استقبال الرسائل
@app.route("/chat", methods=["POST"])
def chat():
    if model is None:
        load_model()

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "No message provided."}), 400

    user_message = data["message"]
    tag, confidence = predict_intent(user_message)

    if confidence > 0.7:
        response = get_response(tag)
    else:
        response = "I'm not sure how to answer that."

    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Read port from environment
    app.run(host="0.0.0.0", port=port)

