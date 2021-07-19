from pre_processing.pre_process import pre_process
from pre_processing.world_tokenize import word_tokenize
from count import count
import pickle
import random

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

def save_model(dictionary_data, name):
    a_file = open(f"{name}.pkl", "wb")
    pickle.dump(dictionary_data, a_file)
    a_file.close()

def load_model(name):
    a_file = open(f"{name}.pkl", "rb")
    output = pickle.load(a_file)
    return output

def create_train_test_set(dataset_name, type):
    file1 = open(f'{dataset_name}', 'r')
    lines = file1.readlines()
    random.shuffle(lines)
    f = open(f"train_set/train.{type}", "a")
    for line in lines[0:int(0.9 * len(lines))]:
        f.write(line)
    f.close()
    f = open(f"test_set/test.{type}", "a")
    for line in lines[int(0.9 * len(lines)):]:
        f.write(line)
    f.close()

def evaluating_model(pos_dict_unigram, neg_dict_unigram, pos_dict_bigram, neg_dict_bigran, λ3, λ2, λ1, ε):
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
    return accuracy

if __name__ == '__main__':
    create = 0
    save = 0
    if create:
        # Creating train and test set
        create_train_test_set('rt-polarity.pos', 'pos')
        create_train_test_set('rt-polarity.neg', 'neg')
    if save:
        # Creating dictionary
        pos_dict_unigram = crete_dict_unigram('train_set/train.pos')
        neg_dict_unigram = crete_dict_unigram('train_set/train.neg')
        pos_dict_bigram  = create_dict_bigram('train_set/train.pos')
        neg_dict_bigram  = create_dict_bigram('train_set/train.neg')
        # Save models
        save_model(pos_dict_unigram,'models/pos_dict_unigram')
        save_model(neg_dict_unigram,'models/neg_dict_unigram')
        save_model(pos_dict_bigram, 'models/pos_dict_bigram')
        save_model(neg_dict_bigram, 'models/neg_dict_bigram')

    pos_dict_unigram = load_model('models/pos_dict_unigram')
    pos_dict_bigram  = load_model('models/pos_dict_bigram')
    neg_dict_unigram = load_model('models/neg_dict_unigram')
    neg_dict_bigram  = load_model('models/neg_dict_bigram')

    print(evaluating_model(pos_dict_unigram,
                           neg_dict_unigram,
                           pos_dict_bigram,
                           neg_dict_bigram,
                           0.75, 0.15, 0.1, 0.1))

    while True:
        text = input()
        res = naive_bayes_classifier(pos_dict_unigram,
                                     neg_dict_unigram,
                                     pos_dict_bigram,
                                     neg_dict_bigram,
                                     text,
                                     0.75, 0.15, 0.1, 0.1)
        if text == '!q':
            break

        if res:
            print('not filter this')
        else:
            print('filter this')