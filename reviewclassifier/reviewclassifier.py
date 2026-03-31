import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

df = pd.read_csv('IMDB Dataset.csv')

df['review'] = df['review'].str.lower()
df['review'] = df['review'].str.replace('!', '')
df['review'] = df['review'].str.replace('.', '')
df['review'] = df['review'].str.replace(',', '')
print(df.head())
count_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = count_vectorizer.fit_transform(df['review'])
Y = df['sentiment'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = SGDClassifier(max_iter=1000, random_state=42)
model.fit(X_train, Y_train)
accuracy = accuracy_score(Y_test, model.predict(X_test))
f1_score = f1_score(Y_test, model.predict(X_test), average='weighted')
confusion_matrix = confusion_matrix(Y_test, model.predict(X_test))
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1_score}')
print(f'Confusion Matrix: {confusion_matrix}')

text = 'film is about cars i would recommend t osee it'
text = text.lower()
text = count_vectorizer.transform([text])
res = model.predict(text)

print(res)
