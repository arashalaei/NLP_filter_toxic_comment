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