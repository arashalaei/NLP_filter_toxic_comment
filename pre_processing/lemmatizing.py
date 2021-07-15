# coding: utf-8
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stopword = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize(text):
    """
    take string input and lemmatize the words.
    use WordNetLemmatizer to lemmatize the words.
    """
    word_tokens = nltk.word_tokenize(text)
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
    return (" ".join(lemmatized_word))