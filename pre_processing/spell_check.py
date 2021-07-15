# coding: utf-8
import nltk
from autocorrect import Speller
from nltk import word_tokenize

def autospell(text):
    """
    correct the spelling of the word.
    """
    s = Speller('en')
    spells = [s.autocorrect_word(w) for w in (nltk.word_tokenize(text))]
    return " ".join(spells)