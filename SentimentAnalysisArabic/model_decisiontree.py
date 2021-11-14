
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils import get_stop_words , clean_str


data_df = pd.read_csv('data/final.csv')
data_df['text'] = data_df.text.apply(lambda x: clean_str(x))
stop_words = r'\b(?:{})\b'.format('|'.join(get_stop_words()))
data_df['text'] = data_df['text'].str.replace(stop_words, '')

# remove the "Neutral" class
data_df = data_df[data_df['sentiment'] != "neutral"]

# change values to numeric
data_df['sentiment'] = data_df['sentiment'].map({'positive': 1, 'negative': 0})

# idneitfy the data and the labels
data = data_df['text']
target = data_df['sentiment']

data_df = data_df.dropna()

# Use TfidfVectorizer for feature extraction (TFIDF to convert textual data to numeric form):
tf_vec = TfidfVectorizer()
X = tf_vec.fit_transform(data)

# Training Phase
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.33, random_state=7)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
"""# DecisionTree Classifier"""
# create the classifer and fit the training data and lables
clf = DecisionTreeClassifier(max_depth=25).fit(X_train, y_train)

# Une fois le modèle ajusté, il est évalué sur l'ensemble de données de test.
accuracy = clf.score(X_test, y_test)
print("RandomForest accuracy: %.2f%%" % (accuracy * 100.0))

with open("models/arabic_sentiment_DecisionTree_tokenizer.pickle", "wb") as f:
    pickle.dump(tf_vec, f)
with open("models/arabic_sentiment_DecisionTree.pickle", "wb") as f:
    pickle.dump(clf, f)

print("Model saved!")