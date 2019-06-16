embedding_dim = 32
hidden_size = 32
batch = 246
cpu_processor = 2
epoch = 2

linear_hidden_size = 32
linear_dropout_keep_prob = 0.1
output_class = 3
learning_rate = 0.001
gpu = 'cuda:0'

preprocessed_train_file = "preprocessed_train_data.pickle"
preprocessed_test_file = "preprocessed_test_data.pickle"
preprocessed_dev_file = "preprocessed_dev_data.pickle"
low_freq_words_file = "low_freq_words_file.txt"
vocab_list = "vocab_list.pickle"
vocab_size = "vocab_size.pickle"
word_to_index = "word_to_index.pickle"
index_to_word = "index_to_word.pickle"
use_stop_word = False
use_remove_freq = False

low_freq = 2

max_seq = 21
min_seq_cut = 3
max_seq_cut = 29
