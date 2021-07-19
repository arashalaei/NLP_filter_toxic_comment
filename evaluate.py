from naive_bayes import naive_bayes_classifier
from unigram_model import unigram_model

def evaluating_model(pos_dict_unigram, neg_dict_unigram, pos_dict_bigram, neg_dict_bigram, λ3, λ2, λ1, ε):
    # confusion Matrix
    CM = {'TP':0, 'FP': 0,
          'FN': 0, 'TN': 0}

    file1 = open('test_set/test.pos', 'r')
    pos_test = file1.readlines()
    file1.close()

    file1 = open('test_set/test.neg', 'r')
    neg_test = file1.readlines()
    file1.close()

    for text in pos_test:
        res = naive_bayes_classifier(pos_dict_unigram,
                                     neg_dict_unigram,
                                     pos_dict_bigram,
                                     neg_dict_bigram,
                                     text,
                                     λ3, λ2, λ1, ε)
        if res:
            CM['TP'] = CM['TP'] + 1
        else:
            CM['FN'] = CM['FN'] + 1

    for text in neg_test:
        res = naive_bayes_classifier(pos_dict_unigram,
                                     neg_dict_unigram,
                                     pos_dict_bigram,
                                     neg_dict_bigram,
                                     text,
                                     λ3, λ2, λ1, ε)
        if res:
            CM['FP'] = CM['FP'] + 1
        else:
            CM['TN'] = CM['TN'] + 1

    accuracy = (CM['TP'] + CM['TN']) / (CM['TP'] + CM['FP'] + CM['FN'] + CM['TN'])
    precision = (CM['TP']) / (CM['TP'] + CM['FP'])
    recall = (CM['TP']) / (CM['TP'] + CM['FN'])
    specificity = (CM['TN']) / (CM['TN'] + CM['FP'])
    f1_score = 2 * ((precision * recall) / (precision + recall))
    return accuracy, precision, recall, specificity, f1_score

def evaluating_unigram(pos_dict_unigram, neg_dict_unigram):
    # confusion Matrix
    CM = {'TP': 0, 'FP': 0,
          'FN': 0, 'TN': 0}

    file1 = open('test_set/test.pos', 'r')
    pos_test = file1.readlines()
    file1.close()

    file1 = open('test_set/test.neg', 'r')
    neg_test = file1.readlines()
    file1.close()

    for text in pos_test:
        res = unigram_model(text, pos_dict_unigram, neg_dict_unigram)
        if res:
            CM['TP'] = CM['TP'] + 1
        else:
            CM['FN'] = CM['FN'] + 1

    for text in neg_test:
        res = unigram_model(text, pos_dict_unigram, neg_dict_unigram)
        if res:
            CM['FP'] = CM['FP'] + 1
        else:
            CM['TN'] = CM['TN'] + 1

    accuracy = (CM['TP'] + CM['TN']) / (CM['TP'] + CM['FP'] + CM['FN'] + CM['TN'])
    precision = (CM['TP']) / (CM['TP'] + CM['FP'])
    recall = (CM['TP']) / (CM['TP'] + CM['FN'])
    specificity = (CM['TN']) / (CM['TN'] + CM['FP'])
    f1_score = 2 * ((precision * recall) / (precision + recall))
    return accuracy, precision, recall, specificity, f1_score
