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
