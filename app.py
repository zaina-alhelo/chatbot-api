
#app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow import keras
import random
import json
import pickle
import os
nltk.data.path.append('./nltk_data')

# Download required NLTK data at startup
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("Successfully loaded NLTK libraries")
except Exception as e:
    print(f"Error loading NLTK libraries: {str(e)}")

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# File paths
INTENTS_PATH = "intents.json"
PICKLE_PATH = "data.pickle"
MODEL_PATH = "model.keras"

# Create Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and data
model = None
words = []
labels = []
intents_data = {}

# Load intents data
def load_intents():
    if not os.path.exists(INTENTS_PATH):
        raise FileNotFoundError(f"{INTENTS_PATH} not found.")
    
    try:
        with open(INTENTS_PATH, encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading intents file: {str(e)}")
        raise

# Prepare data for training
def preprocess_data(data):
    words, labels, docs_x, docs_y = [], [], [], []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # Tokenize words
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        
        # Collect unique tags
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Lemmatize words and remove duplicates
    words = sorted(set(lemmatizer.lemmatize(w.lower()) for w in words if w != "?"))
    labels = sorted(labels)

    # Create training data
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

# Load or prepare the model
def load_model():
    global model, words, labels, intents_data
    
    print("Starting model loading/preparation process...")
    
    try:
        # Load intents data
        intents_data = load_intents()
        print("Intents data loaded successfully")
        
        # Check if preprocessed data exists
        if os.path.exists(PICKLE_PATH):
            print("Loading preprocessed data from pickle file...")
            with open(PICKLE_PATH, "rb") as f:
                words, labels, training, output = pickle.load(f)
            print("Preprocessed data loaded successfully")
        else:
            print("Preprocessing training data...")
            training, output, words, labels = preprocess_data(intents_data)
            print("Saving preprocessed data...")
            with open(PICKLE_PATH, "wb") as f:
                pickle.dump((words, labels, training, output), f)
            print("Preprocessed data saved successfully")
        
        # Check if model exists
        if os.path.exists(MODEL_PATH):
            print("Loading existing model...")
            model = keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully")
        else:
            print("Creating and training new model...")
            model = keras.models.Sequential([
                keras.layers.Dense(256, activation='relu', input_shape=(len(training[0]),)),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(len(output[0]), activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(training, output, epochs=600, batch_size=16, verbose=1)
            print("Saving new model...")
            model.save(MODEL_PATH)
            print("Model saved successfully")
        
        return model, words, labels, intents_data
        
    except Exception as e:
        print(f"Error in load_model function: {str(e)}")
        raise

# Convert input to bag of words
def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# Predict intent from sentence
def predict_intent(sentence):
    global model, words, labels
    
    bow = bag_of_words(sentence, words)
    results = model.predict(np.array([bow]))[0]
    results_index = np.argmax(results)
    tag = labels[results_index]
    confidence = results[results_index]
    return tag, confidence

# Get response based on the predicted intent
def get_response(tag):
    global intents_data
    
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I don't understand."

# Load model at startup
try:
    print("Loading model at application startup...")
    model, words, labels, intents_data = load_model()
    print("Model successfully loaded at startup")
except Exception as e:
    print(f"Error loading model at startup: {str(e)}")
    # Continue without model - it will attempt to load on first request

# Home page
@app.route("/", methods=["GET"])
def home():
    return "🤖 Chatbot API is running!"

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    global model, words, labels, intents_data
    
    try:
        # Check if model is loaded
        if model is None:
            print("Model not loaded, attempting to load now...")
            model, words, labels, intents_data = load_model()
        
        # Get and validate request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        if "message" not in data:
            return jsonify({"error": "No message provided"}), 400
        
        user_message = data["message"]
        print(f"Received message: '{user_message}'")
        
        # Predict intent and get response
        tag, confidence = predict_intent(user_message)
        print(f"Predicted tag: '{tag}' with confidence: {confidence}")
        
        if confidence > 0.7:
            response = get_response(tag)
        else:
            response = "I'm not sure how to answer that."
            
        print(f"Sending response: '{response}'")
        return jsonify({"response": response})
        
    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)