import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gluonnlp.data import PadSequence
from config import FLAGS

class Corpus():
    def __init__(self, vocab, tokenizer):
        self._vocab = vocab
        self._tokenizer = tokenizer

    def token2idx(self, item):
        label_dict = {"neutral": 0, "entailment": 1, "contradiction": 2, "-": 3}
        sen1, sen2, label = tf.io.decode_csv(item, record_defaults=[[""],[""],[""]], field_delim='\t')

        sen1 = [self._tokenizer.morphs(sen.numpy().decode('utf-8')) for sen in sen1]
        sen1 = [[self._vocab.token_to_idx[token] for token in sen] for sen in sen1]
        sen1 = pad_sequences(sen1, maxlen=FLAGS.length, value=0,
                             padding='post', truncating='post')

        sen1 = tf.convert_to_tensor(sen1, dtype=tf.float32)

        sen2 = [self._tokenizer.morphs(sen.numpy().decode('utf-8')) for sen in sen2]
        sen2 = [[self._vocab.token_to_idx[token] for token in sen] for sen in sen2]
        sen2 = pad_sequences(sen2, maxlen=FLAGS.length, value=0,
                             padding='post', truncating='post')
        sen2 = tf.convert_to_tensor(sen2, dtype=tf.float32)

        label = tf.convert_to_tensor([label_dict[l.numpy().decode('utf-8')] for l in label], dtype=tf.float32)
        label = tf.reshape(label, [FLAGS.batch_size,1])
        return sen1, sen2, label
