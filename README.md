# Arabic Tweet Hashtag Sentiment Analysis

## Modules 
### Data Used for training

The data used is a public dataset of tagged arabic tweets. There is 60k instances of data.
The data is in the `data` folder.
### LSTM model

The model training and definition is in the `lstm_model.py` file. 
This can be run to get the LSTM model and tokenizer frozen to the `models` folder. 
he model is a simple model with an `Embedding` Layer, an `LSTM` Layer and a final 
`sigmoid` layer for prediciting the probabilities. The optimizer used is `adam` and 
loss function is the `binary_crossentropy`. The `Tokenizer` in the keras library is used.

### SVM Model
The model training and definition is in the `svm_model.py` file.
The model takes in input from the `TFIDF Vectorizer` and uses the `SVM.SVC` classifier.
The model is frozen to the `models` folder along with the vectorizer for inference.

### DataCleaner

This module uses `nltk` package for various text cleaning and preprocessing. It removes stopwords, stemming etc.

### Model Inference

This handles loading the models and tokenizer into memory and then doing prediction with them.
The `model_inference.py` has the code for the same.
The `get_sentiment()` in this module takes in a dataframe and the model with which to predict.

### Tweet Manager

This module takes care of authentication to the twitter API and calls to the Twitter search endpoint.
The `config.ini` file can be edited to add authentication credentials to do the same. 
The `get_tweets()` function takes in the query, result type, count and the language of the tweets to be parsed.
The file `tweetmanager.py` has the code for the same.

### Web App

The web app definition is in the `SA_app.py`. It uses streamlit library for defining the web app flow. Model loading 
and tweet scraping is cached so that it is not repeated each time causing the UI to freeze.


## Instructions to run

In the project directory run the following to set up the required libraries

`pip install -r requirements.txt`

To run the app

`streamlit run SA_app.py`

This will start the app in localhost and in a given port displayed in the commandline.
