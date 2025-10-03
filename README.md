Emotion Classifier NLP Project
 Summary:
I built a small NLP project that predicts emotions from text. I cleaned and explored the dataset, trained and compared two models (Logistic Regression and Naive Bayes), and deployed a Flask API for real-time predictions. This demonstrates my skills in data preprocessing, machine learning, model evaluation, and API deployment. The project is simple, but end-to-end, showing practical understanding of an ML workflow.
 
The workflow includes:  
1. Loading and cleaning text data  
2. Exploring the dataset  
3. Training two models (Logistic Regression and Naive Bayes)  
4. Evaluating models using accuracy, precision, recall, F1-score, and confusion matrices  
5. Making predictions via a CLI script or a Flask API  

## Dataset

- Source: [Kaggle Emotion Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)  
- Number of samples: 16,000 (training set)  
- Categories: `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`  

The dataset is publicly available and contains sentences labeled with multiple emotions. 

## Project Structure
emotion-classifier/
│
├─ data/  Contains train.txt, test.txt, val.txt
├─ src/
│ ├─ preprocess.py  Text cleaning and preprocessing
│ ├─ explore.py  Exploratory analysis and plots
│ ├─ train.py  Model training and evaluation
│ ├─ predict.py  CLI script to predict emotion for input text
│ └─ app.py  Flask API for emotion prediction
├─ venv/  Python virtual environment
├─ model.pkl  Saved Logistic Regression model
├─ vectorizer.pkl  Saved TF-IDF vectorizer
└─ README.md  This file


1. Install dependencies

Make sure your virtual environment is active:

pip install -r requirements.txt
Required libraries: pandas, nltk, scikit-learn, matplotlib, seaborn, flask, joblib, requests

2. Preprocess & Explore Data
python src/explore.py
This will show:

-Total samples and per-emotion counts
-Most common words in the dataset
-Bar plots for label distribution and word frequencies

3. Train Models
python src/train.py

-Splits data into train and test sets
-Trains Logistic Regression and Naive Bayes models
-Evaluates models with precision, recall, F1-score, and confusion matrices
-Saves the best model (model.pkl) and TF-IDF vectorizer (vectorizer.pkl)

Model Comparison:
-logistic Regression: Accuracy 0.79 – better balanced across emotions
-Naive Bayes: Accuracy 0.66 – less consistent

4. Predict Using CLI
python src/predict.py
Enter any sentence to see the predicted emotion
Type exit to quit

5. Predict Using Flask API
python src/app.py

Runs the API at: http://127.0.0.1:5000
Example request using Postman or curl:
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"text":"I feel amazing today!"}'
Sample Response:
{
  "text": "I feel amazing today!",
  "predicted_emotion": "joy"
}

Python test script:
python src/test_api.py

Features Implemented
Data Cleaning: Lowercase, remove punctuation, remove stopwords
Exploratory Analysis: Show dataset statistics and most common words
Model Training: Logistic Regression and Naive Bayes with TF-IDF vectorizer
Evaluation: Classification reports and confusion matrices
Prediction: CLI and Flask API
Bonus: TF-IDF vectorization, model comparison, confusion matrix visualization


Developer: Ardolina Ziberi
GitHub: github.com/yourusername/emotion-classifier
