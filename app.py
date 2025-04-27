import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model as keras_load_model
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

app = Flask(__name__)

# Paths
PICKLE_PATH = "data.pickle"
MODEL_PATH = "model.h5"
INTENTS_PATH = "intents.json"

# Initialize ChatterBot
chatbot = ChatBot(
    'EyeBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///chatterbot.sqlite3'
)

list_trainer = ListTrainer(chatbot)

# Function to train ChatterBot
def train_chatterbot():
    if intents_data:
        for intent in intents_data["intents"]:
            for pattern in intent['patterns']:
                for response in intent['responses']:
                    list_trainer.train([pattern, response])

# Global variables
model = None
words = []
labels = []
training = []
output = []
intents_data = None

# Load intents from JSON file
def load_intents():
    global intents_data
    with open(INTENTS_PATH) as file:
        intents_data = json.load(file)

# Preprocess intents data
def preprocess_data(intents_data):
    words = []
    labels = []
    training_sentences = []
    training_labels = []
    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            word_list = pattern.split()
            words.extend(word_list)
            training_sentences.append(pattern)
            training_labels.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = sorted(set(words))
    labels = sorted(set(labels))

    training = []
    output = []
    for sentence in training_sentences:
        bag = [0] * len(words)
        word_list = sentence.split()
        for word in word_list:
            if word in words:
                bag[words.index(word)] = 1
        training.append(bag)

    for tag in training_labels:
        output_row = [0] * len(labels)
        output_row[labels.index(tag)] = 1
        output.append(output_row)

    return training, output, words, labels

# Load model and prepare training
def load_model():
    global model, words, labels, training, output, intents_data

    intents_data = load_intents()

    if os.path.exists(PICKLE_PATH):
        with open(PICKLE_PATH, "rb") as f:
            words, labels, training, output = pickle.load(f)
    else:
        training, output, words, labels = preprocess_data(intents_data)
        with open(PICKLE_PATH, "wb") as f:
            pickle.dump((words, labels, training, output), f)

    if os.path.exists(MODEL_PATH):
        model = keras_load_model(MODEL_PATH)
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

    train_chatterbot()  # Train ChatterBot

# Function to predict intent using model
def predict_intent(sentence):
    bag = [0] * len(words)
    word_list = sentence.split()
    for word in word_list:
        if word in words:
            bag[words.index(word)] = 1
    prediction = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(prediction) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list[0]["intent"], float(return_list[0]["probability"])

# Function to get response from intents
def get_response(tag):
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])
    return "Sorry, I did not understand that."

# Route to handle chat
@app.route("/chat", methods=["POST"])
def chat():
    if model is None:
        load_model()

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "No message provided."}), 400

    user_message = data["message"]
    tag, confidence = predict_intent(user_message)

    # If confidence is high, use model response
    if confidence > 0.7:
        response = get_response(tag)
    else:
        # If confidence is low, fallback to ChatterBot
        response = str(chatbot.get_response(user_message))

    return jsonify({
        "response": response
    })

# Start the server
if __name__ == "__main__":
    load_model()
    app.run(debug=True)
