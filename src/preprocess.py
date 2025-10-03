import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            text, label = parts
            texts.append(text)
            # only keeping the emotion part (after ";")
            labels.append(label.split(";")[-1])
    df = pd.DataFrame({"text": texts, "label": labels})
    df["clean_text"] = df["text"].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_data("data/train.txt")  
    print(df.head())
    print("Number of samples:", len(df))
    print("Categories:\n", df["label"].value_counts())


