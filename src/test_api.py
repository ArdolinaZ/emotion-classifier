
import requests

url = "http://127.0.0.1:5000/predict"

sentences = [
    "I feel amazing today!",
    "I am very sad right now",
    "I am so angry at this"
]

print("Testing Flask API...\n")

for sentence in sentences:
    response = requests.post(url, json={"text": sentence})
    if response.status_code == 200:
        data = response.json()
        print(f"Input: {data['text']}")
        print(f"Predicted Emotion: {data['predicted_emotion']}\n")
    else:
        print(f"Error: {response.status_code} - {response.text}\n")

print("Testing complete!")
