embedding_dim = 32
hidden_size = 32
batch = 256
cpu_processor = 2
epoch = 5

linear_hidden_size = 32
linear_dropout_keep_prob = 0.1
output_class = 3
learning_rate = 0.0005
gpu = 'cuda:0'

folder = "preprocessing/"
preprocessed_train_file = folder + "preprocessed_train_data.pickle"
preprocessed_test_file = folder + "preprocessed_test_data.pickle"
preprocessed_dev_file = folder + "preprocessed_dev_data.pickle"
low_freq_words_file = folder + "low_freq_words_file.txt"
vocab_list = folder + "vocab_list.pickle"
vocab_size = folder + "vocab_size.pickle"
word_to_index = folder + "word_to_index.pickle"
index_to_word = folder + "index_to_word.pickle"
use_stop_word = False
use_UNK = False
use_remove_low_freq = False
create_vocab_file_list = True

low_freq = 2

max_seq = 21
min_seq_cut = 3
max_seq_cut = 29

path = "../../data/snli_1.0/"
path_train = path + "snli_1.0_train.txt"
path_test = path + "snli_1.0_test.txt"
path_dev = path + "snli_1.0_dev.txt"

labels = ["neutral", "contradiction", "entailment"] 
UNK = "[UNK]"
PAD = "[PAD]"

train_mode = 0
test_mode = 1 
dev_mode = 2
all_mode = 3