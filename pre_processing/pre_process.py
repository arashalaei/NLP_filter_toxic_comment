from pre_processing.contractions_dict import expand_contractions, contractions_dict
from pre_processing.lemmatizing import lemmatize
from pre_processing.remove_numbers import remove_numbers
from pre_processing.remove_punctutaions import remove_punct
from pre_processing.remove_tags import remove_Tags
from pre_processing.spell_check import autospell
from pre_processing.stemming import stemming
from pre_processing.stopword_removal import remove_stopwords
from pre_processing.to_lower import to_lower
from pre_processing.world_tokenize import word_tokenize


def pre_process(text):
    """
    """
    text = expand_contractions(text, contractions_dict)
    text = autospell(text)
    text = to_lower(text)
    text = remove_numbers(text)
    text = remove_punct(text)
    text = remove_Tags(text)
    text = lemmatize(text)
    text = stemming(text)
    text = remove_stopwords(text)
    # text = word_tokenize(text)
    return text
