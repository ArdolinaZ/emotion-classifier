
# Emotion Classifier Project

## Goal
Build a small AI/ML project to classify emotions in text. The project demonstrates:
- Loading and preprocessing text data
- Exploratory data analysis
- Training and evaluating models
- Making predictions on new text

This project is designed to show end-to-end workflow for a text classification task using Python and scikit-learn.

---

## Dataset
- **Source:** [Kaggle Emotion Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
- Contains multiple emotions such as: joy, sadness, anger, fear, love, surprise
- Training samples: 16,000+  
- Validation and test sets included  
- Minimum 2–5 categories: ✅

---

## Project Structure

emotion-classifier/
├─ data/ # Raw dataset files (train.txt, test.txt, val.txt)
├─ src/ # Source code
│ ├─ preprocess.py # Data cleaning and preprocessing
│ ├─ explore.py # Exploratory analysis and statistics
│ ├─ train.py # Model training, evaluation, and saving models
│ ├─ predict.py # Script to predict emotion for new sentences
│ └─ app.py #  Flask API for predictions
├─ venv/ # Python virtual environment
├─ model.pkl # Saved trained model (Logistic Regression)
├─ vectorizer.pkl # Saved TF-IDF vectorizer
└─ README.md # Project description


---

## Features Implemented

1. **Data Preparation**
   - Loaded and inspected dataset
   - Cleaned text (lowercase, remove punctuation, remove stopwords)
   - Added `clean_text` column for preprocessed sentences

2. **Exploratory Analysis**
   - Checked number of samples per emotion category
   - Identified most frequent words per category (optional visualization)
   - Ensured balanced dataset for training

3. **Model Training**
   - Used **TF-IDF vectorizer** for feature extraction
   - Compared **Logistic Regression** and **Naive Bayes**
   - Split dataset into training and test sets
   - Evaluated models with **accuracy, precision, recall, F1-score**
   - Saved the best model and vectorizer for predictions

4. **Prediction**
   - Script `predict.py` accepts a sentence and outputs the predicted emotion
   - Example usage:

```python
from predict import predict_emotion

sentence = "I feel amazing today!"
predicted_emotion = predict_emotion(sentence)
print(predicted_emotion)  # Output: joy

How to Run

Create virtual environment and activate:

python -m venv venv
venv\Scripts\activate


Install required packages:

pip install -r requirements.txt


Preprocess data:

python src\preprocess.py


Explore data (optional):

python src\explore.py


Train model:

python src\train.py


Predict new sentences:

python src\predict.py

Model Performance
Model	Accuracy	Notes
Logistic Regression	79%	Best performing
Naive Bayes	66%	Lower accuracy, good baseline

Classification reports and confusion matrices are printed during training.

Optional Bonus Features (Not Required)

TF-IDF vectorization used instead of raw bag-of-words

Compared two models to select the best one

Flask API (app.py) available for serving predictions

Confusion matrix visualization available in explore.py

Conclusion

This project demonstrates the full workflow for a text classification task:

Data cleaning & preprocessing

Exploratory data analysis

Model training & evaluation

Making predictions on new sentences

It is suitable for showcasing ML skills and Python proficiency in an internship application.
