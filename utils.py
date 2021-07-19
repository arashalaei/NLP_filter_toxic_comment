import pickle
import random


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