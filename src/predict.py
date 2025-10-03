import joblib
from preprocess import clean_text

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to predict the emotion of a sentence
def predict_emotion(text):
    cleaned_text = clean_text(text)
    
    # Convert text to features using the vectorizer
    features = vectorizer.transform([cleaned_text])
    
    # Predict the emotion
    predicted_emotion = model.predict(features)[0]
    
    return predicted_emotion


if __name__ == "__main__":
    print("Welcome to the Emotion Predictor!")
    print("Type a sentence and see its predicted emotion.")
    print("Type 'exit' to quit.\n")
    
    while True:
        sentence = input("Enter a sentence: ")
        
        # Exit condition
        if sentence.lower() == "exit":
            print("Goodbye!")
            break
        
        # Predict emotion
        prediction = predict_emotion(sentence)
        
        print("Predicted Emotion:", prediction, "\n")

