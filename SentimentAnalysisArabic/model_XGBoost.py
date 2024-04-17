
import pickle
import re
import codecs
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def get_stop_words():
    path = "data/stop_words.txt"
    stop_words = []
    with codecs.open(path, "r", encoding="utf-8", errors="ignore") as myfile:
        stop_words = myfile.readlines()
    stop_words = [word.strip() for word in stop_words]
    return stop_words

# Nettoyer / normaliser le texte arabe
def clean_str(text):

    # retirer At'tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)

    # retirer longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    # retirer les caractères doublons
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    #Remplacer les caractères non desiré
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
              '&quot;', '?', '؟', '!','\ufeff','#','  ','   ','--']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ',
               ' ! ' , '','',' ',' ',' ']
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # trim pour retirer les espaces blancs avant et arrière les textes
    text = text.strip()

    return text






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
"""# XGBoost Classifier"""
# create the classifer and fit the training data and lables
clf = XGBClassifier(max_depth=25, n_estimators=400).fit(X_train, y_train)

# make predictions for test data
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))

with open("models/arabic_sentiment_XGBoost_tokenizer.pickle", "wb") as f:
    pickle.dump(tf_vec, f)
with open("models/arabic_sentiment_XGBoost.pickle", "wb") as f:
    pickle.dump(clf, f)

print("Model saved!")