import base64

import keras.backend.tensorflow_backend as tb
import pandas as pd
import streamlit as st

tb._SYMBOLIC_SCOPE.value = True

from model_inference import Inference
from tweetmanger import TweetManager
from results import Results


@st.cache(allow_output_mutation=True)
def load_model():
    """
     Charger le modèle de classification et le mettre en cache
    """
    with st.spinner('Loading classification model...'):
        classifier = Inference()

    return classifier


@st.cache(allow_output_mutation=True)
def init_twitter():
    """
    Charger l'API Twitter et le mettre en cache
    """
    with st.spinner('Loading Twitter Manager...'):
        tweet_manager = TweetManager()

    return tweet_manager


@st.cache(allow_output_mutation=True)
def get_twitter_data(tweet_manager, tweet_input, sidebar_result_type, sidebar_tweet_count):
    """
    Obtenir des données à partir de l'API Twitter et du cache
    """
    df = tweet_manager.get_tweets(tweet_input, result_type=sidebar_result_type, count=sidebar_tweet_count)

    return df


def get_table_download_link(df):
    """Génère un lien permettant de télécharger les données d'un dataframe
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download as csv file (Right click and save link as csv)</a>'
    return href


def main():
    """
    Fonction principale appelée qui permet d'afficher l'interface utilisateur
    """

    # La définition du titre
    st.title("Sentiment Analyzer on Twitter Hashtags in Arabic Language")

    # Les définitions de la barre à gauche pour configurer l'application
    st.sidebar.header("Model options")

    # La liste de sélection pour choisir le type du modele
    sidebar_model_type = st.sidebar.selectbox('Model Type', ('LSTM', 'SVM', 'CNN','DecisionTree','naivebayes','RandomForest','XGBoost'), index=2)

    # Tweet parser options.
    st.sidebar.header("Tweet Parsing options")

    # Curseur pour configurer le nombre de tweets à analyser.
    sidebar_tweet_count = st.sidebar.slider(label='Number of tweets',
                                            min_value=50,
                                            max_value=5000,
                                            value=50,
                                            step=50)

    # Type de tweets à analyser, à ne pas modifier.
    sidebar_result_type = st.sidebar.selectbox('Result Type', ('popular', 'mixed', 'recent'), index=1)

    pd.set_option('display.max_colwidth', 0)

    # Les modèles sont chargés et conservés en mémoire pour des performances optimisées.
    classifier = load_model()

    # L'authentification à l'API Twitter est effectuée et conservée en mémoire.
    tweet_manager = init_twitter()

    # Classe pour afficher les résultats
    results = Results()

    # Zone de saisie pour analyser le hashtag
    st.subheader('Input the hashtag to analyze')
    tweet_input = st.text_input('Hashtag:')

    if tweet_input != '':
        # Obtenir les tweets
        with st.spinner('Parsing from twitter API'):
            # L'API Twitter est appelé
            df = get_twitter_data(tweet_manager, tweet_input, sidebar_result_type, sidebar_tweet_count)

        # st.dataframe(df)


        # Faire les prédictions
        if df.__len__() > 0:
            with st.spinner('Predicting...'):
                # Si les tweets sont présents, la prédiction est effectuée sur le dataframe.
                pred = classifier.get_sentiment(df, sidebar_model_type)

                # Prédictions et le dataframe sont affichées.
                st.subheader('Prediction:')
                st.dataframe(pred)

                # Lien de téléchargement pour le fichier csv contenant les prédictions.
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)

            # Tous les résultats sont calculés
            results.calculate_results(df)

            # Mettre toutes les visualisations dans l'interface utilisateur
            with st.spinner('Generating Visualizations...'):
                st.header('Visualizations')
                st.subheader('Pie Chart') # Diagramme circulaire
                st.plotly_chart(results.get_pie_chart_counts())
                st.subheader('Bar Chart') # diagramme à barres
                st.plotly_chart(results.get_bar_chart_counts())
                st.subheader('Pie Chart showing most used words') # Graphique à secteurs montrant les mots les plus utilisés
                st.plotly_chart(results.get_pie_chart_most_counts())
                st.subheader('Bar Chart showing most used words') # Graphique à barres montrant les mots les plus utilisés
                st.plotly_chart(results.get_bar_chart_most_counts())

                st.subheader('Time series showing tweets') # Séries chronologiques des tweets
                st.plotly_chart(results.get_line_chart_tweets())

                st.header('Tables') # Visualisation des tableaux
                st.subheader('Hashtag Analysis')
                st.write('Number of tweets per classification') # Nombre de tweets par classification
                st.table(results.get_stats_table())

                st.subheader('Most words frequency') # Fréquence des mots
                st.write('Top 15 words from tweets') # les 15 premiers mots des tweetsles plus utilisés
                st.table(results.get_stats_table_most_counts())

                st.subheader('Word cloud') # Nuage de mots
                st.write('Top 50 words in all the tweets represented as word cloud') # Les 50 premiers mots les plus utilisés de tous les tweets représentés sous forme de nuage de mots
                results.get_word_cloud()
                st.pyplot()

                st.write('Top 30 words in positive tweets') # Les 30 premiers mots les plus utilisés dans les tweets positifs
                results.get_word_cloud_positive()
                st.pyplot()

                st.write('Top 30 words in negative tweets') # Les 30 premiers mots les plus utilisés dans les tweets négatifs
                results.get_word_cloud_negative()
                st.pyplot()


        else:
            # Si aucun tweet trouvé
            st.write('No tweets found')


if __name__ == '__main__':
    main()
