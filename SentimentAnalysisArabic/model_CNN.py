import codecs
import pickle
import re

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D ,Flatten
from tensorflow.keras.layers.embeddings import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import plot_model

MAX_NUM_WORDS = 40000
VALIDATION_SPLIT = 0.2

# Fonction pour recupérer les stopwords
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

    # retirer longation (”جمييييييييييل ” devient ” جمييل ”). 
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    # retirer les caractères doublons
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    #Remplacer les caractères non desiré
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
              '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ',
               ' ! ']
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # trim pour retirer les espaces blancs avant et arrière les textes
    text = text.strip()

    return text




# Charger le dataset
df = pd.read_csv("data/final.csv")
# Nettoyer et supprimer les stop_words
df['text'] = df.text.apply(lambda x: clean_str(x))
stop_words = r'\b(?:{})\b'.format('|'.join(get_stop_words()))
df['text'] = df['text'].str.replace(stop_words, '')
# Affecter la valeur 1 pour les tweets positives et 0 pour les negatifs
df['binary_sentiment'] = df.sentiment.map(dict(positive=1, negative=0))
# Faire un mélange des données
df = shuffle(df)


# Partitionner le dataset en 20% pour le test et 80% pour l'apprentisage
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['binary_sentiment'], test_size=0.20, random_state=42)

'''
L'étape suivante consiste à coder les données sous la forme d'une séquence d'entiers.
car le modele nécessite des entrées entières où chaque entier correspond à un seul mot 
qui a une représentation vectorielle  à valeur réelle. 
'''
# créer le tokenizer
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
# adapter le tokenizer sur les données
tokenizer.fit_on_texts(X_train)

'''
Maintenant que le mappage des mots en nombres entiers a été préparé, nous pouvons l'utiliser pour encoder 
les tweets dans le jeu de données d'apprentissage.
'''
# encodage de séquence
sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

'''
utiliser pad_sequences pour assurer que tous les données ont la même longueur.
Ou on limite les séquences à la longueur maximale qui est 100.
'''
# pad sequences
data = pad_sequences(sequences, maxlen=100)
test_data = pad_sequences(test_sequences, maxlen=100)


# définir la taille maximal des mots (la plus grande valeur entière)
vocab_size = len(tokenizer.word_index) + 1 #72089


# Model defnition
model_cnn = Sequential()
model_cnn.add(Embedding(vocab_size, 100, input_length=100))
model_cnn.add(Conv1D(padding="same", kernel_size=8, filters=32, activation="relu"))
model_cnn.add(MaxPooling1D(pool_size=2)) # réduit de moitié le rendement de la couche (Conv1D).
model_cnn.add(Flatten())
model_cnn.add(Dense(250, activation='relu'))
model_cnn.add(Dense(1, activation='sigmoid'))
# Resumé du model
print(model_cnn.summary())
plot_model(model_cnn, show_shapes=True, to_file='model.png')

'''
Ensuite, nous adaptons le réseau aux données d'apprentisage.
-La fonction 'binary_crossentropy' parce qu'on a un problème de classification binaire. 
-L'implémentation  d'Adam de la descente de gradient  
-Nous gardons une trace de la précision(accuracy) en plus de la perte(loss) pendant le training. 
-Le modèle est formé pendant 10 epochs, càd on va passer 10 fois à travers les données d'apprentisage.
'''

# compiler le réseau
model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit le réseau
history = model_cnn.fit(data, y_train, validation_split=VALIDATION_SPLIT, epochs=10, batch_size = 128)

'''
Une fois le modèle ajusté, il est évalué sur l'ensemble de données de test.
'''
# évaluer le modèle
loss, acc = model_cnn.evaluate(test_data, y_test, verbose=0)
print("Accuracy de test (CNN) est : {0:.2f} %".format(acc*100))



# Enregistrer le model sur le disque
with open("models/arabic_sentiment_cnn2.pickle", "wb") as f:
    pickle.dump(tokenizer, f)
model_cnn.save('models/arabic_sentiment_cnn2.hdf5')
print("Model saved in disk")


# Fonction pour Afficher un résume de l’historique lors de l’apprentissage du model CNN sous forme d'un graphe

def plot_history(model):
    acc = model.history['accuracy']
    val_acc = model.history['val_accuracy']
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, "b", label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


plot_history(history)
