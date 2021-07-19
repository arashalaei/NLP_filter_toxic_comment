from pre_processing.pre_process import pre_process
from pre_processing.world_tokenize import word_tokenize


def count_unigram(dict, token):
    try:
        return dict[token]
    except KeyError:
        return 0


def unigram_model(text, pos_dict_unigram, neg_dict_unigram):
    text = pre_process(text)
    tokens = word_tokenize(text)

    p_pos = 1
    p_neg = 1
    M_pos = len(pos_dict_unigram)
    M_neg = len(neg_dict_unigram)
    v = len(tokens)

    for token in tokens:
        p_pos *= ((1 + count_unigram(pos_dict_unigram,token)) / (M_pos + v))
        p_neg *= ((1 + count_unigram(neg_dict_unigram,token)) / (M_neg + v))

    if p_pos > p_neg:
        return 1
    else:
        return 0
