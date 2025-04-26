from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import os
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Import after NLTK downloads to avoid errors
try:
    from mainc import predict_intent, get_responses, chatbot, words, labels, model, load_data
    intents_data = load_data("intents.json")
except Exception as e:
    print(f"Error importing modules: {e}")

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API is running",
        "message": "Welcome to the chatbot API!",
        "available_endpoints": ["/chat"]
    })

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        try:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "Invalid JSON or no content provided"}), 400
            
            user_input = data.get("message")
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
                return jsonify({"reply": "Sorry, I'm having trouble answering right now."})
        
        except Exception as e:
            print("Error in /chat route:", e)
            return jsonify({"error": "An error occurred. Please try again later."}), 500
    
    elif request.method == "GET":
        return jsonify({
            "message": "Please send a POST request to this endpoint with JSON data: {'message': 'your question here'}"
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)