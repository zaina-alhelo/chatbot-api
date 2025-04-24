from flask import Flask, request, jsonify
from flask_cors import CORS
from mainc import predict_intent, get_responses, chatbot, words, labels, model, load_data
import random

app = Flask(__name__)
CORS(app)

intents_data = load_data("intents.json")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.get_json(silent=True).get("message")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        if model:
            tag, confidence = predict_intent(user_input, words, labels, model)
            print(f"[Model] Tag: {tag}, Confidence: {confidence}")
            if confidence > 0.7:
                responses = get_responses(tag, intents_data)
                if responses:
                    return jsonify({"reply": random.choice(responses)})

        print("Low confidence, using ChatterBot...")
        try:
            response = chatbot.get_response(user_input)
            return jsonify({"reply": str(response)})
        except Exception as e:
            print("ChatterBot error:", e)
            return jsonify({"reply": "Sorry, I’m having trouble answering right now."})

    except Exception as e:
        print("Error in /chat route:", e)
        return jsonify({"error": "An error occurred. Please try again later."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
