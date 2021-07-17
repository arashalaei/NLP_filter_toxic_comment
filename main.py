from pre_processing.pre_process import pre_process
from pre_processing.world_tokenize import word_tokenize
from count import count

def crete_dict_unigram(file_name):
    file1 = open(file_name, 'r')
    lines = file1.readlines()
    l = []
    for line in lines:
        text = pre_process(line)
        word_token = word_tokenize(text)
        l += word_token
    return count(l)

def create_dict_bigram(file_name):
    file1 = open(file_name, 'r')
    lines = file1.readlines()
    test_list = []
    for line in lines:
        text = pre_process(line)
        test_list.append(text)
    res = [(x, i.split()[j + 1]) for i in test_list for j, x in enumerate(i.split()) if j < len(i.split()) - 1]
    return count(res)

# P(wi) = count(wi) / M
def probability_unigram(w, uni_dict, M):
    try:
        return uni_dict[w] / M
    except KeyError:
        return 0

# P(wi|wi-1) = count(wi-1 wi) / count(wi-1)
def probability_bigram(w, uni_dict, bi_dict):
    try:
        return bi_dict[w] / uni_dict[w[0]]
    except KeyError:
        return 0

# P(wi|wi-1) = λ3 P(wi|wi-1) + λ2 P(wi) + λ1 ε
def interpolation(λ3, λ2, λ1, ε, w, uni_dict, bi_dict, M):
    return (λ3 * probability_bigram(w, uni_dict, bi_dict)) + (λ2 * probability_unigram(w[1], uni_dict, M)) + (λ1 * ε)

def naive_bayes_classifier(pos_dict_unigram, neg_dict_unigram, pos_dict_bigram, neg_dict_bigran, text, λ3, λ2, λ1, ε):
    text = pre_process(text)
    print(text)
    word_token = word_tokenize(text)
    print(word_token)
    test_list = [text]
    pairs = [(x, i.split()[j + 1]) for i in test_list for j, x in enumerate(i.split()) if j < len(i.split()) - 1]
    print(pairs)
    M_pos = len(pos_dict_unigram)
    M_neg = len(neg_dict_unigram)
    p_pos = probability_unigram(word_token[0],pos_dict_unigram,M_pos) # p of positive class
    p_neg = probability_unigram(word_token[0],neg_dict_unigram,M_neg) # p of negative class
    for i in pairs:
        p_pos *= interpolation(λ3, λ2, λ1, ε, i, pos_dict_unigram, pos_dict_bigram, M_pos)
        p_pos *= interpolation(λ3, λ2, λ1, ε, i, neg_dict_unigram, neg_dict_bigram, M_neg)

    print(p_pos)
    print(p_neg)
    if(p_pos > p_neg):
        return 1
    else:
        return 0

if __name__ == '__main__':
    # Creating dictionary
    pos_dict_unigram = crete_dict_unigram('rt-polarity.pos')
    neg_dict_unigram = crete_dict_unigram('rt-polarity.neg')
    pos_dict_bigram = create_dict_bigram('rt-polarity.pos')
    neg_dict_bigram = create_dict_bigram('rt-polarity.neg')

    text = """with or without ballast tanks , k-19 sinks to a harrison ford low ."""

    res = naive_bayes_classifier(pos_dict_unigram, neg_dict_unigram, pos_dict_bigram, neg_dict_bigram, text, 0.15, 0.75, 0.1, 0.0000001)
    print(res)