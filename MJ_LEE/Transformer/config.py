import re

## hyper-parameter
epoch = 10
batch = 64
learning_rate = 0.0005

embedding_dim = 300
hidden_size = 128
linear_hidden_size = 128

## option
remove_outlier_seq = True
show_statistics = False # 전처리 할 때만 볼 수 있음
use_glove = True
vocab_mode = 3 # all_mode

low_freq = 2
max_seq = 21
min_seq_cut = 3
max_seq_cut = 29

## fixed
cpu_processor = 2
linear_dropout_keep_prob = 0.1
output_class = 3
gpu = 'cuda:0'
folder = "preprocessing/"
vocab_list = folder + "vocab_list.json"
word_to_index = folder + "word_to_index.json"
glove_json = folder + "glove_json.json"
path = "../../data/snli_1.0/"
path_train = path + "snli_1.0_train.txt"
path_test = path + "snli_1.0_test.txt"
path_dev = path + "snli_1.0_dev.txt"
glove = "../../data/glove.840B.300d.txt"
glove_npy = folder + "glove.npy"
labels = ["neutral", "contradiction", "entailment"] 
UNK = "[UNK]"
PAD = "[PAD]"

train_mode = 0
test_mode = 1 
dev_mode = 2
all_mode = 3

regex = re.compile(r"[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'\_…》]")