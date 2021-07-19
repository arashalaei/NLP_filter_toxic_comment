from pre_processing.pre_process import pre_process
from pre_processing.world_tokenize import word_tokenize
from compute_probabillity import probability_unigram,  interpolation


def naive_bayes_classifier(pos_dict_unigram, neg_dict_unigram, pos_dict_bigram, neg_dict_bigram, text, λ3, λ2, λ1, ε):
    text = pre_process(text)
    word_token = word_tokenize(text)
    test_list = [text]
    pairs = [(x, i.split()[j + 1]) for i in test_list for j, x in enumerate(i.split()) if j < len(i.split()) - 1]
    M_pos = len(pos_dict_unigram)
    M_neg = len(neg_dict_unigram)
    p_pos = probability_unigram(word_token[0],pos_dict_unigram,M_pos) # p of positive class
    p_neg = probability_unigram(word_token[0],neg_dict_unigram,M_neg) # p of negative class

    for i in pairs:
        p_pos *= interpolation(λ3, λ2, λ1, ε, i, pos_dict_unigram, pos_dict_bigram, M_pos)
        p_neg *= interpolation(λ3, λ2, λ1, ε, i, neg_dict_unigram, neg_dict_bigram, M_neg)

    if(p_pos > p_neg):
        return 1
    else:
        return 0