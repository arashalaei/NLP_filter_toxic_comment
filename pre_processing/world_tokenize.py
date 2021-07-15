import nltk


def word_tokenize(text):
    """
    take string input and return list of words.
    use nltk.word_tokenize() to split the words.
    """
    word_list = []
    for sentences in nltk.sent_tokenize(text):
        for words in nltk.word_tokenize(sentences):
            word_list.append(words)
    return word_list


