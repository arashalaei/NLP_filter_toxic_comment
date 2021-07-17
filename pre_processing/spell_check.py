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


# import pip
#
#
# def install(package):
#     if hasattr(pip, 'main'):
#         pip.main(['install', package])
#     else:
#         pip._internal.main(['install', package])
#
#     # Example
#
#
# if __name__ == '__main__':
#     install('pandas')