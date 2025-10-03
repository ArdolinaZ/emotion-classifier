from preprocess import load_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

TRAIN_PATH = "data/train.txt"

print("Loading data...")
df = load_data(TRAIN_PATH)

# Clean labels (keep only the emotion after ';')
df['label'] = df['label'].apply(lambda x: x.split(';')[-1])

print("Data loaded! Sample categories:\n", df['label'].value_counts())

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB()
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_vec, y_train)
    
    print("Predicting on test set...")
    y_pred = model.predict(X_test_vec)
    
   
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

joblib.dump(models["Logistic Regression"], "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nTraining complete! Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'.")
