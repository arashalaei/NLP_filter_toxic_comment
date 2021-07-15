def count(pre_process_text):
    word_count = dict()

    for word in pre_process_text:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    return word_count