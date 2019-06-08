import tensorflow as tf
import pandas as pd

from gluonnlp.data import PadSequence
from gluonnlp import Vocab
from mecab import MeCab


class Corpus():
    def __init__(self, vocab: Vocab, tokenizer: MeCab):
        self._vocab = vocab
        self._toknizer = tokenizer

    def token2idex(self, item):
        sen1, sen2, gold_label = tf.io.decode_csv(item, record_defaults=[[''],[0, 1, 2]], field_delim='\t')
        print(sen1)
        print(sen2)
        print(gold_label)

        # self._corpus = pd.read_csv(item, sep='\t').iloc[:, [0, 1, 2]]
        # label_dict = {"neutral": 0, "entailment": 1, "contradiction": 2, "-": 3}
        #
        # try:
        #     sen = self._toknizer.morphs(self._corpus.iloc[idx][0])
        # except:
        #     sen = self._toknizer.morphs('')
        # sen2indices = tf.convert_to_tensor(self._padder([self._vocab.token_to_idx[token] for token in sen]), dtype=tf.long)
        #
        # try:
        #     sen2 = self._toknizer.morphs(self._corpus.iloc[idx][1])
        # except:
        #     sen2 = self._toknizer.morphs('')
        # sen22indices = tf.convert_to_tensor(self._padder([self._vocab.token_to_idx[token] for token in sen2]), dtype=tf.long)
        #
        # gold_label = tf.convert_to_tensor(label_dict[self._corpus.iloc[idx][2]], dtype=tf.long)

        #return gold_label, sen2indices, sen22indices
