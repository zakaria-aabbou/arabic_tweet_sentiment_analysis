import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

class DataCleaner:
    cleaned_data = None

    def __init__(self):
        return

    def tokenize(self, txt):
        tokens = word_tokenize(txt)
        return tokens

    def remove_stop_words(self, tokens):
        stops = stopwords.words('arabic')
        valuable_words = []
        for word in tokens:
            if word not in stops:
                valuable_words.append(word)
        sentence = " ".join(valuable_words)
        return sentence

    def remove_stops(self, tokens):
        stops = stopwords.words('arabic')
        valuable_words = []
        for word in tokens:
            if word not in stops:
                valuable_words.append(word)

        return valuable_words

    # def remove_tags(self):
    #     return
    #
    # def remove_hashtags(self):
    #     return
    #
    # def remove_special_chars(self):
    #     return

    def clean(self, txt):
        clean = re.sub(r'(?is)[-_]', " ", str(txt))
        clean = re.sub(r'(?is)[^أ-ي ❤☻☺]', '', str(clean))
        clean = re.sub("[إأٱآا]", "ا", clean)
        clean = re.sub("[إأٱآا]+", 'ا', clean)
        #clean = re.sub("ى", "ي", clean)
        #clean = re.sub("ؤ", "ء", clean)
        #clean = re.sub("ة", "ه", clean)
        #clean = re.sub("ئ", "ء", clean)
        noise = re.compile(""" ّ 
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)
        clean = re.sub(noise, '', clean)
        return clean

    # def remove_foreign_words(self):
    #     return
    #
    # def remove_numbers(self):
    #     return

    def analyze_emotions(self, txt):
        emotions = re.sub("❤", "حب", txt)
        emotions = re.sub("☺☻", "ضحك", emotions)
        return emotions

    def stem(self, txt):
        st = ISRIStemmer()
        stem_words = []
        words = self.tokenize(txt)
        for w in words:
            stem_words.append(st.stem(w))

        return stem_words

    def prepare_data_set(self, data):

        sentence = []

        for key, value in data.items():
            text = self.clean(key)
            text = self.tokenize(text)
            text = self.remove_stop_words(text)
            text = self.analyze_emotions(text)
            text = self.stem(text)
            sentence.append((text, value))

        return sentence

    def prepare_data_list(self, data):

        sentence = []

        for key in data:
            text = self.clean(key)
            text = self.tokenize(text)
            text = self.remove_stop_words(text)
            text = self.analyze_emotions(text)
            text = self.stem(text)
            sentence.append(' '.join(text))

        return sentence

    def prepare_data_set_without_stem(self, data):

        sentence = []

        for key in data:
            text = self.clean(key)
            text = self.tokenize(text)
            text = self.remove_stops(text)
            for w in text:
                sentence.append(w)

        return sentence
