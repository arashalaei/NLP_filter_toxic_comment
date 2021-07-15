from pre_processing.pre_process import pre_process
from pre_processing.world_tokenize import word_tokenize
from count import count

if __name__ == '__main__':
    text = pre_process("""Similaly, once you'd beyond this surprising first step, 
                            you'd realise that Satanism isn't at all like what the movies have shown us. 
                            Whereas Harry Potter couldn't wait to put his muggle past behind him, Sabrina 
                            is having a more difficult time making up her mind. At one point, Sabrina and her 
                            friends even make a slight reference to Riverdale high school, perhaps teasing a future 
                            crossover. Kiernan Shipka is off to the Academy of Unseen Arts in Chilling Adventures of 
                            Sabrina. With minor adjustments The Chilling Adventures of Sabrina could be Netflix's new 
                            A Series of Unfortunate Events, but with Umbrella Academy in the offing, the streaming 
                            service's offbeat YA material seems to be cannibalising itself.""")
    word_token = word_tokenize(text)
    test_list = [text]
    res = [(x, i.split()[j + 1]) for i in test_list for j, x in enumerate(i.split()) if j < len(i.split()) - 1]

    # Unigram
    print(count(word_token))
    # Bigram
    print(count(res))