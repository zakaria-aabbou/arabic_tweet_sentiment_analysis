
import re
import codecs
import pandas as pd


'''
Definir ici tous les fonctions de nettoyage des textes arabes
'''
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
              '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ',
               ' ! ']
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # trim pour retirer les espaces blancs avant et arrière les textes
    text = text.strip()

    return text


if __name__ == '__main__':
    
    df = pd.read_csv('data/final.csv')
    df['text'] = df.text.apply(lambda x: clean_str(x))
    stop_words = r'\b(?:{})\b'.format('|'.join(get_stop_words()))
    df['text'] = df['text'].str.replace(stop_words, '')
    print(df['text'])