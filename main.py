from pre_processing.pre_process import pre_process
from pre_processing.world_tokenize import word_tokenize
from count import count
from create_dict import create_dict_bigram, crete_dict_unigram
from compute_probabillity import probability_unigram, probability_bigram, interpolation
from naive_bayes import naive_bayes_classifier
from evaluate import evaluating_model, evaluating_unigram
from unigram_model import unigram_model
from utils import save_model, load_model, create_train_test_set


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

    λ3, λ2, λ1, ε = 0.75, 0.15, 0.1, 0.1
    eval = evaluating_model(pos_dict_unigram,
                           neg_dict_unigram,
                           pos_dict_bigram,
                           neg_dict_bigram,
                           λ3, λ2, λ1, ε)

    print('*** Result of evaluating bigram model***')
    print(f'F1-Score: {eval[4]}')
    print(f'Accuracy: {eval[0]}')
    print(f'Precision: {eval[1]}')
    print(f'Recall: {eval[2]}')
    print(f'specificity: {eval[3]}')
    print('#####################################')

    eval = evaluating_unigram(pos_dict_unigram,neg_dict_unigram)
    print('*** Result of evaluating unigram model***')
    print(f'F1-Score: {eval[4]}')
    print(f'Accuracy: {eval[0]}')
    print(f'Precision: {eval[1]}')
    print(f'Recall: {eval[2]}')
    print(f'specificity: {eval[3]}')
    print('#####################################')
    while True:
        text = input()
        res = naive_bayes_classifier(pos_dict_unigram,
                                     neg_dict_unigram,
                                     pos_dict_bigram,
                                     neg_dict_bigram,
                                     text,
                                     λ3, λ2, λ1, ε)
        if text == '!q':
            break

        if res:
            print('not filter this')
        else:
            print('filter this')