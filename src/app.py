from flask import Flask, request, jsonify
import joblib
from preprocess import clean_text

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from the request
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "Please provide some text to predict."}), 400

    clean = clean_text(text)
    features = vectorizer.transform([clean])

    prediction = model.predict(features)[0]

    return jsonify({"text": text, "predicted_emotion": prediction})

if __name__ == "__main__":
    # Start Flask server in debug mode
    app.run(debug=True)

