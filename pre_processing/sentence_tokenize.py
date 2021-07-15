# coding: utf-8
import nltk
from nltk import word_tokenize


def sentence_tokenize(text):
    """
    take string input and return list of sentences.
    use nltk.sent_tokenize() to split the sentences.
    """
    sent_list = []
    for w in nltk.sent_tokenize(text):
        sent_list.append(w)
    return sent_list