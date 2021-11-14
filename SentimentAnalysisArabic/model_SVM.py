
import pickle
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix , classification_report


from utils import get_stop_words , clean_str


# Charger le dataset
data_df = pd.read_csv('data/final.csv')
# Nettoyer et supprimer les stop_words
data_df['text'] = data_df.text.apply(lambda x: clean_str(x))
stop_words = r'\b(?:{})\b'.format('|'.join(get_stop_words()))
data_df['text'] = data_df['text'].str.replace(stop_words, '')



# supprimer la classe "Neutre"
data_df = data_df[data_df['sentiment'] != "neutral"]

# changer les valeurs en numérique
data_df['sentiment'] = data_df['sentiment'].map({'positive': 1, 'negative': 0})

# identifier les données et les labels
data = data_df['text']
target = data_df['sentiment']

data_df = data_df.dropna()

# Utilisez TfidfVectorizer pour l'extraction des fonctionnalités 
#(TFIDF pour convertir les données textuelles sous forme numérique):
tf_vec = TfidfVectorizer()
X = tf_vec.fit_transform(data)

# Partitionner le dataset en 33% pour le test et 67% pour l'apprentisage
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.33, random_state=7)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
"""# SVM Classifier"""
# créer le classificateur et ajuster les données et les labels d'apprentissage
classifier_svm = svm.SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)

# Une fois le modèle ajusté, il est évalué sur l'ensemble de données de test.
accuracy = classifier_svm.score(X_test, y_test)
print("SVM accuracy: %.2f%%" % (accuracy * 100.0))


with open("models/arabic_sentiment_svm_tokenizer.pickle", "wb") as f:
    pickle.dump(tf_vec, f)
with open("models/arabic_sentiment_svm.pickle", "wb") as f:
    pickle.dump(classifier_svm, f)
