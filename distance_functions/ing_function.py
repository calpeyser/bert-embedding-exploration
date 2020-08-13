import os

def vocab_file():
    this_file = os.path.dirname(__file__)
    vocab_file = os.path.join(this_file, '../small_bert_bert_uncased_L-2_H-128_A-2_1/assets/vocab.txt')
    return vocab_file

def read_vocab():
    print(vocab_file())
    f = open(vocab_file(), 'rb')
    words = f.readlines()
    words = [word.rstrip() for word in words]
    return words

def has_ing():
    vocab = read_vocab()
    return [s.endswith('ing') for s in vocab]

class IngLabelFunction(object):

    def __init__(self):
        super(IngLabelFunction, self).__init__()
        self.lookup = has_ing()

    def label(self, id):
        return self.lookup[id]

