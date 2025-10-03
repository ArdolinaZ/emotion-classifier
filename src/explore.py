from preprocess import load_data
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_PATH = "data/train.txt"


df = load_data(TRAIN_PATH)


df['label'] = df['label'].apply(lambda x: x.split(';')[-1])

print("Total samples:", len(df))


print("\nSamples per emotion:")
print(df['label'].value_counts())


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="label", order=df['label'].value_counts().index, palette="Set2")
plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.show()


all_words = " ".join(df['clean_text']).split()
common_words = Counter(all_words).most_common(15)

print("\nMost common words:")
for word, freq in common_words:
    print(f"{word}: {freq}")


words, freqs = zip(*common_words)
plt.figure(figsize=(8, 5))
sns.barplot(x=list(freqs), y=list(words), palette="viridis")
plt.title("Most Common Words in Dataset")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()
