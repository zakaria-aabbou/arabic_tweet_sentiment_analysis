import codecs
import pickle
import re

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers.embeddings import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

MAX_NUM_WORDS = 40000
VALIDATION_SPLIT = 0.2


def get_stop_words():
    path = "data/stop_words.txt"
    stop_words = []
    with codecs.open(path, "r", encoding="utf-8", errors="ignore") as myfile:
        stop_words = myfile.readlines()
    stop_words = [word.strip() for word in stop_words]
    return stop_words


def get_text_sequences(texts):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=100)
    return data, word_index


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



df = pd.read_csv("data/final.csv")
## Clean and drop stop words
df['text'] = df.text.apply(lambda x: clean_str(x))
stop_words = r'\b(?:{})\b'.format('|'.join(get_stop_words()))
df['text'] = df['text'].str.replace(stop_words, '')
df['binary_sentiment'] = df.sentiment.map(dict(positive=1, negative=0))
df = shuffle(df)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['binary_sentiment'], test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)
data = pad_sequences(sequences, maxlen=100)
test_data = pad_sequences(test_sequences, maxlen=100)

# définir la taille du vocabulaire (la plus grande valeur entière)
vocab_size = len(tokenizer.word_index) + 1 #72089

# Model defnition
model_lstm = Sequential()
model_lstm.add(Embedding(vocab_size, 100, input_length=100))
model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model_lstm.fit(data, y_train, validation_split=VALIDATION_SPLIT, epochs=10, batch_size = 128)

loss, acc = model_lstm.evaluate(test_data, y_test, verbose=0)
print("Accuracy de test (LSTM) est : {0:.2f} %".format(acc*100))


with open("models/arabic_sentiment_lstm2.pickle", "wb") as f:
    pickle.dump(tokenizer, f)
model_lstm.save('models/arabic_sentiment_lstm2.hdf5')


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

plot_history(history)
