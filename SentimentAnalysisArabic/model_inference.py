import pickle

from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


class Inference(object):

    def __init__(self):
        """
        Les modèles sont préchargés afin que cela ne prenne pas de temps lors de l'inférence
        """
        #CNN
        self.cnn_model = load_model('models/arabic_sentiment_cnn.hdf5')
        with  open("models/arabic_sentiment_cnn.pickle", "rb") as f:
            self.cnn_pickle = pickle.load(f)

        #LSTM
        self.lstm_model = load_model('models/arabic_sentiment_lstm.hdf5')
        with  open("models/arabic_sentiment_lstm.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)

        #SVM
        with  open("models/arabic_sentiment_svm.pickle", "rb") as f:
            self.svm_model = pickle.load(f)
        with  open("models/arabic_sentiment_svm_tokenizer.pickle", "rb") as f:
            self.svm_tfidf = pickle.load(f)

        #DecisionTree
        with  open("models/arabic_sentiment_DecisionTree.pickle", "rb") as f:
            self.DecisionTree_model = pickle.load(f)
        with  open("models/arabic_sentiment_DecisionTree_tokenizer.pickle", "rb") as f:
            self.DecisionTree_tfidf = pickle.load(f)

        #naivebayes
        with  open("models/arabic_sentiment_naivebayes.pickle", "rb") as f:
            self.naivebayes_model = pickle.load(f)
        with  open("models/arabic_sentiment_naivebayes_tokenizer.pickle", "rb") as f:
            self.naivebayes_tfidf = pickle.load(f)

        #RandomForest
        with  open("models/arabic_sentiment_RandomForest.pickle", "rb") as f:
            self.RandomForest_model = pickle.load(f)
        with  open("models/arabic_sentiment_RandomForest_tokenizer.pickle", "rb") as f:
            self.RandomForest_tfidf = pickle.load(f)

        #XGBoost
        with  open("models/arabic_sentiment_XGBoost.pickle", "rb") as f:
            self.XGBoost_model = pickle.load(f)
        with  open("models/arabic_sentiment_XGBoost_tokenizer.pickle", "rb") as f:
            self.XGBoost_tfidf = pickle.load(f)


    def get_sentiment(self, df, model):
        """
        Prend une entrée de texte sur laquelle vous souhaitez exécuter une analyse des sentiments.
        Retourne le score de sentiment et la classe de sentiment (positif ou négatif)

        :param text_input: Text to run sentiment analysis on
        :return: (sentiment_score, sentiment_class)
        """

        if model == 'LSTM':
            sequences = self.tokenizer.texts_to_sequences(df['tweet'])
            data = pad_sequences(sequences, maxlen=100)
            num_class = self.lstm_model.predict(data)
            df['sentiment_score'] = num_class
        elif model == 'SVM':
            data = df['tweet']
            X = self.svm_tfidf.transform(data)
            num_class = self.svm_model.predict_proba(X)
            df['sentiment_score'] = [num[1] for num in num_class]
        elif model == 'CNN':
            sequences = self.cnn_pickle.texts_to_sequences(df['tweet'])
            data = pad_sequences(sequences, maxlen=100)
            num_class = self.cnn_model.predict(data)
            df['sentiment_score'] = num_class
        elif model == 'DecisionTree':
            data = df['tweet']
            X = self.DecisionTree_tfidf.transform(data)
            num_class = self.DecisionTree_model.predict_proba(X)
            df['sentiment_score'] = [num[1] for num in num_class]
        elif model == 'naivebayes':
            data = df['tweet']
            X = self.naivebayes_tfidf.transform(data)
            num_class = self.naivebayes_model.predict_proba(X)
            df['sentiment_score'] = [num[1] for num in num_class]
        elif model == 'RandomForest':
            data = df['tweet']
            X = self.RandomForest_tfidf.transform(data)
            num_class = self.RandomForest_model.predict_proba(X)
            df['sentiment_score'] = [num[1] for num in num_class]
        elif model == 'XGBoost':
            data = df['tweet']
            X = self.XGBoost_tfidf.transform(data)
            num_class = self.XGBoost_model.predict_proba(X)
            df['sentiment_score'] = [num[1] for num in num_class]

        def score_segregate(value):
            if value <= 0.35:
                return 'Negative'
            elif value > 0.35 and value < 0.65:
                return 'Neutral'
            elif value >= 0.65:
                return 'Positive'

        df['sentiment_class'] = df['sentiment_score'].apply(score_segregate)

        return df


def main():
    """
    Pour tester le classificateur
    """
    import pandas as pd
    df = Inference().get_sentiment(pd.read_csv('Files/corona.csv'), 'RandomForest')
    print(df)


if __name__ == '__main__':
    main()
