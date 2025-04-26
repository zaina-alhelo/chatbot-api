from flask import Flask, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow import keras
import random
import json
import pickle
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# تحميل ملفات NLTK المطلوبة
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

INTENTS_PATH = "intents.json"
PICKLE_PATH = "data.pickle"
MODEL_PATH = "model.keras"
CHATTERBOT_DB = "chatterbot.sqlite3"

app = Flask(__name__)  # << هنا عرفنا كائن app

# تحميل وتجهيز النموذج
model, words, labels, intents_data = None, None, None, None

def load_data(file_path):
    with open(file_path, encoding='utf-8') as file:
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

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w != "?"]
    words = sorted(set(words))
    labels = sorted(labels)

    training, output = [], []
    out_empty = [0 for _ in range(len(labels))]

    for i, doc in enumerate(docs_x):
        bag = [1 if w in [lemmatizer.lemmatize(word.lower()) for word in doc] else 0 for w in words]
        output_row = out_empty.copy()
        output_row[labels.index(docs_y[i])] = 1
        training.append(bag)
        output.append(output_row)

    return np.array(training), np.array(output), words, labels

def prepare_model():
    data = load_data(INTENTS_PATH)
    try:
        with open(PICKLE_PATH, "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:
        training, output, words, labels = preprocess_data(data)
        with open(PICKLE_PATH, "wb") as f:
            pickle.dump((words, labels, training, output), f)

    try:
        model = keras.models.load_model(MODEL_PATH)
    except:
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

    return model, words, labels, data

def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def predict_intent(sentence, words, labels, model):
    bow = bag_of_words(sentence, words)
    results = model.predict(np.array([bow]))[0]
    results_index = np.argmax(results)
    tag = labels[results_index]
    confidence = results[results_index]
    return tag, confidence

def get_response(tag, intents_data):
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I don't understand."

@app.route("/")
def home():
    return "Bot is up and running!"

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    global model, words, labels, intents_data
    if not model:
        model, words, labels, intents_data = prepare_model()

    data = request.get_json()
    user_message = data.get("message")

    if user_message:
        tag, confidence = predict_intent(user_message, words, labels, model)
        if confidence > 0.7:
            response = get_response(tag, intents_data)
            return jsonify({"response": response})
        else:
            return jsonify({"response": "I'm not sure how to answer that."})
    else:
        return jsonify({"response": "No message provided."})

if __name__ == "__main__":
    app.run(debug=True)
