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

# تحميل ملفات NLTK المطلوبة
nltk.download('punkt')
nltk.download('wordnet')

# تهيئة lemmatizer
lemmatizer = WordNetLemmatizer()

# مسارات الملفات
INTENTS_PATH = "intents.json"
PICKLE_PATH = "data.pickle"
MODEL_PATH = "model.keras"
CHATTERBOT_DB = "chatterbot.sqlite3"

# تحميل البيانات من ملف JSON
def load_data(file_path):
    try:
        with open(file_path, encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# معالجة البيانات وتجهيزها للتدريب
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

# تحضير النموذج والبيانات
def prepare_model():
    data = load_data(INTENTS_PATH)
    if data:
        try:
            with open(PICKLE_PATH, "rb") as f:
                words, labels, training, output = pickle.load(f)
        except (FileNotFoundError, EOFError):
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
    else:
        return None, None, None, None

# تحويل الجملة إلى bag of words
def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# التنبؤ بالنية (intent)
def predict_intent(sentence, words, labels, model):
    bow = bag_of_words(sentence, words)
    results = model.predict(np.array([bow]))[0]
    results_index = np.argmax(results)
    tag = labels[results_index]
    confidence = results[results_index]
    return tag, confidence

# جلب الردود الخاصة بالنية المتوقعة
def get_response(tag, intents_data):
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I don't understand."

# إعداد ChatterBot
def prepare_chatterbot():
    chatbot = ChatBot(
        'EyeBot',
        storage_adapter='chatterbot.storage.SQLStorageAdapter',
        database_uri=f'sqlite:///{CHATTERBOT_DB}'
    )

    list_trainer = ListTrainer(chatbot)

    data = load_data(INTENTS_PATH)
    if data:
        for intent in data['intents']:
            for pattern in intent['patterns']:
                for response in intent['responses']:
                    list_trainer.train([pattern, response])

    return chatbot

# وظيفة المحادثة
def chat():
    model, words, labels, intents_data = prepare_model()
    chatbot = prepare_chatterbot()

    if not model:
        print("Model not available. Exiting.")
        return

    print("Bot is ready to chat! Type 'quit' to exit.")

    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            print("Goodbye!")
            break

        tag, confidence = predict_intent(inp, words, labels, model)
        if confidence > 0.7:
            response = get_response(tag, intents_data)
            print(f"Bot (Intent): {response}")
        else:
            try:
                response = chatbot.get_response(inp)
                print(f"Bot (General): {response}")
            except Exception as e:
                print(f"Bot (Error): {e}")
                print("Bot: Sorry, I couldn't understand that.")

# تشغيل البرنامج
if __name__ == "__main__":
    chat()
