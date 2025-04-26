import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow import keras
import random
import json
import pickle
from time import sleep
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import spacy
import os

# تحميل الموارد اللازمة لمكتبة NLTK
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# تحميل النموذج الخاص بـ SpaCy
try:
    spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")

# Paths for necessary files
INTENTS_PATH = "intents.json"
PICKLE_PATH = "data.pickle"
MODEL_PATH = "model.keras"
CHATTERBOT_DB = "chatterbot.sqlite3"

def load_data(file_path):
    try:
        with open(file_path) as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def preprocess_data(data):
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for i, doc in enumerate(docs_x):
        bag = []
        wrds = [lemmatizer.lemmatize(w.lower()) for w in doc]
        for w in words:
            bag.append(1) if w in wrds else bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[i])] = 1
        training.append(bag)
        output.append(output_row)

    return np.array(training), np.array(output), words, labels

# Load or process data
data = load_data(INTENTS_PATH)
if data:
    try:
        with open(PICKLE_PATH, "rb") as f:
            words, labels, training, output = pickle.load(f)
    except (FileNotFoundError, EOFError):
        training, output, words, labels = preprocess_data(data)
        with open(PICKLE_PATH, "wb") as f:
            pickle.dump((words, labels, training, output), f)
    
    # Build the TensorFlow 2.x model using tf.keras
    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
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
else:
    print("Data not loaded. Please check the intents.json file.")
    model = None

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def predict_intent(input_sentence, words, labels, model):
    bow = bag_of_words(input_sentence, words)
    results = model.predict(np.array([bow]))[0]
    results_index = np.argmax(results)
    tag = labels[results_index]
    confidence = results[results_index]
    return tag, confidence

def get_responses(tag, intents_data):
    for intent in intents_data["intents"]:
        if intent['tag'] == tag:
            return intent['responses']
    return ["I'm sorry, I don't have a response for that."]

# Initialize ChatterBot
chatbot = ChatBot(
    'EyeBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri=f'sqlite:///{CHATTERBOT_DB}'
)

# Create ListTrainer for ChatterBot
list_trainer = ListTrainer(chatbot)

# Train ChatterBot with intents
if data:
    for intent in data['intents']:
        for pattern in intent['patterns']:
            for response in intent['responses']:
                list_trainer.train([pattern, response])

def chat():
    print("Hi, How can I help you!")
    intents_data = load_data(INTENTS_PATH)
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        if model:
            tag, confidence = predict_intent(inp, words, labels, model)
            print(f"Intent Model - Predicted tag: {tag}, Confidence: {confidence}")

            if confidence > 0.7:
                responses = get_responses(tag, intents_data)
                if responses:
                    response = random.choice(responses)
                    sleep(0.5)
                    print("Bot (Intent):", response)
                else:
                    print("Bot (Intent): No response found for this intent.")
            else:
                print("Intent confidence is low. Asking general chatbot...")
                try:
                    response = chatbot.get_response(inp)
                    sleep(0.5)
                    print("Bot (General):", response)
                except Exception as e:
                    print(f"ChatterBot Error: {e}")
                    print("Bot (General): Sorry, I'm having trouble responding generally right now.")

if __name__ == "__main__":
    chat()
