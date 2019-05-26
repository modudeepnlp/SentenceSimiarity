#%%
import re
import torch

class Custom_dataset():
    def __init__(self):
        # path = "./Ma_LSTM/snli_1.0/snli_1.0/"
        path = "./snli_1.0/snli_1.0/"
        self.path_train = path + "snli_1.0_train.txt"
        self.path_test = path + "snli_1.0_train.txt"
        self.path_dev = path + "snli_1.0_train.txt"

        self.max_length = 30
        self.UNK = "[UNK]"
        self.PAD = "[PAD]"

        self.labels = ["neutral", "contradiction", "entailment"] 
        self.label_to_index = {
            self.labels[0] : 0,  
            self.labels[1] : 1, 
            self.labels[2] : 2
            }

        self.index_label = {
            0 : self.labels[0],  
            1 : self.labels[1], 
            2 : self.labels[2]
        }

    def __regular_expresion(self, sentence):
        clear = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sentence)
        return clear

    def __indexing(self, sentences):
        all_sentences = ' '.join(sentence for sentence in sentences)
        all_words = all_sentences.split(' ')
        
        self.no_duplication_words = [self.UNK, self.PAD]
        self.no_duplication_words.extend(list(set(all_words)))

        self.word_to_index = { word:i for i,word in enumerate(self.no_duplication_words) }
        self.index_to_word = { i:word for i,word in enumerate(self.no_duplication_words) }
        self.vocab_size = len(self.word_to_index)

    def __UNK(self, word_array):
        for n, word in enumerate(word_array):
            if word not in self.no_duplication_words:
                word_array[n] = self.UNK

        return word_array

    def __txt_read(self, path, is_train):
        # string 작업
        if is_train:
            print("--Start train--")
        else:
            print("--Start test--")

        str_train_data = []
        all = []
        with open(path, "r") as f:
            for n, line in enumerate(f):
                if n == 0:
                    continue
                parts = line.strip().split("\t")
                gold_label = self.__regular_expresion(parts[0]) 
                sentence1 = self.__regular_expresion(parts[5]) 
                sentence2 = self.__regular_expresion(parts[6]) 
                word_array1 = sentence1.split(' ')
                word_array2 = sentence2.split(' ')
                all.append(sentence1)
                all.append(sentence2)

                str_train_data.append((gold_label, word_array1, word_array2))
        print("--Done to word array--")
     
        # indexing, UNK 작업, (PAD 작업이 필요 없는것 같음.. pytorch 에서는)
        if is_train == True:
            self.__indexing(all)
        else :
            word_array1 = self.__UNK(word_array1)
            word_array2 = self.__UNK(word_array2)

        index_train_data = []
        for data_tuple in str_train_data:
            (gold_label, word_array1, word_array2) = data_tuple
            if gold_label not in self.labels: # 레이벨 없는것 예외처리
                continue
            index_label = self.label_to_index[gold_label]
            index_sent1 = [self.word_to_index[word] for word in word_array1]
            index_sent2 = [self.word_to_index[word] for word in word_array2]
            index_train_data.append((index_label, index_sent1, index_sent2))
        
        print("--Done Indexing, UNK, Max_Length--")
        return index_train_data

    def get_data(self):
        self.index_train_data = self.__txt_read(self.path_train, True)
        self.index_test_data = self.__txt_read(self.path_test, False)
        # self.index_dev_data = self.__txt_read(self.path_test, False)

        return (self.index_train_data, self.index_test_data)

