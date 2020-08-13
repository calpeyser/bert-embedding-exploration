import os
import tensorflow.compat.v1 as tf

def vocab_file():
    vocab_file = os.path.join(os.path.abspath(''), 'small_bert_bert_uncased_L-2_H-128_A-2_1/assets/vocab.txt')
    return vocab_file

def read_vocab():
    f = open(vocab_file(), 'rb')
    words = f.readlines()
    words = [word.rstrip() for word in words]
    return words

def has_ing():
    vocab = read_vocab()
    return [s.endswith('ing') for s in vocab]

class IngDistanceFunction(object):

    def __init__(self):
        super(IngDistanceFunction, self).__init__()
        self.lookup = tf.constant(has_ing())

    def is_ing(self, ids):
        ings = tf.gather(self.lookup, ids)
        ings = tf.cast(ings, dtype=tf.float32)
        return ings

    def distance(self, ids_1, ids_2):
        ings_1 = tf.gather(self.lookup, ids_1)
        ings_2 = tf.gather(self.lookup, ids_2)
        are_different = tf.logical_xor(ings_1, ings_2)
        distance = tf.cast(are_different, dtype=tf.int32)
        return distance

ids1 = [
    [9836, 0, 0, 0, 0],
    [0, 9836, 0, 0, 0],
    [0, 0, 9836, 0, 0],
]
ids1 = tf.constant(ids1)

ids2 = [
    [0, 0, 0, 0, 9836],
    [0, 9836, 0, 9836, 0],
    [0, 0, 9836, 0, 0],
]
ids2 = tf.constant(ids2)

dist_func = IngDistanceFunction()
res = dist_func.is_ing(ids1)

with tf.Session() as sess:
    print(res)
    res_val = sess.run(res)
    print(res_val)