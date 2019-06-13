import re
import torch
import numpy as np
import config as config
from torch.nn.utils.rnn import pad_sequence

class Custom_dataset():
    def __init__(self):
        path = "../../data/snli_1.0/"
        self.path_train = path + "snli_1.0_train.txt"
        self.path_test = path + "snli_1.0_test.txt"
        self.path_dev = path + "snli_1.0_dev.txt"

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
        return clear.lower()

    def __indexing_vocab(self, sentences):
        all_sentences = ' '.join(sentence for sentence in sentences)
        all_words = all_sentences.split(' ')
        
        self.no_duplication_words = [self.UNK, self.PAD] # PAD Index = 1
        self.no_duplication_words.extend(list(set(all_words)))

        self.word_to_index = { word:i for i,word in enumerate(self.no_duplication_words) }
        self.index_to_word = { i:word for i,word in enumerate(self.no_duplication_words) }
        self.vocab_size = len(self.word_to_index)

    def __find_unk(self, word):
        if [word] not in self.no_duplication_words: # 찾는 형태가 배열이여야지 찾음
            return self.UNK
        else:
            return word

    def __UNK(self, str_train_data):
        new_str_train_data = []
        for gold_label, word_array1, word_array2 in str_train_data:
            new_word_array1 = [self.__find_unk(word) for word in word_array1]
            new_word_array2 = [self.__find_unk(word) for word in word_array2]   
            new_str_train_data.append((gold_label, new_word_array1, new_word_array2))
        return new_str_train_data

    def __PAD(self, index_array):
        while len(index_array) < config.max_seq:
            index_array.append(1) # PAD_INDEX
        return index_array

    def __Max_length(self, index_array):
        if len(index_array) > config.max_seq:
            index_array = index_array[:30]
        return index_array

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
                parts = line.strip().split("\t") # 한 줄당 여러 탭으로 파트가 나뉘어 져 있음
                gold_label = self.__regular_expresion(parts[0]) 
                sentence1 = self.__regular_expresion(parts[5]) 
                sentence2 = self.__regular_expresion(parts[6]) 
                word_array1 = sentence1.split(' ')
                word_array2 = sentence2.split(' ')
                all.append(sentence1)
                all.append(sentence2)

                str_train_data.append((gold_label, word_array1, word_array2))
        print("--Done sentence to word array--")
     
        # indexing, UNK 작업, (PAD 작업이 필요 없는것 같음.. pytorch 에서는)
        if is_train == True:
            # word index list는 train 에서만 만든다.
            self.__indexing_vocab(all)
        else :
            # train 에는 UNK는 없다. (train과 test 는 다른 분포라고 가정)
            str_train_data = self.__UNK(str_train_data)

        # train index로 word 를 index 전환 
        index_train_data = []
        for n, data_tuple in enumerate(str_train_data):
            (gold_label, word_array1, word_array2) = data_tuple
            if gold_label not in self.labels: # 레이벨 없는것 예외처리
                continue
            index_label = self.label_to_index[gold_label]
            index_sent1 = [self.word_to_index[word] for word in word_array1]
            index_sent2 = [self.word_to_index[word] for word in word_array2]
            
            # max_len 모자란 길이 PAD 입력
            index_sent1 = self.__PAD(index_sent1)
            index_sent2 = self.__PAD(index_sent2)

            # max_len 넘는 배열 제거
            index_sent1 = self.__Max_length(index_sent1)
            index_sent2 = self.__Max_length(index_sent2)

            index_train_data.append((index_label, index_sent1, index_sent2))
        
        print("--Done Indexing, UNK, PAD, Max_Length")
        return index_train_data

    def get_data(self):
        self.index_train_data = self.__txt_read(self.path_train, True)
        self.index_test_data = self.__txt_read(self.path_test, False)
        # self.index_dev_data = self.__txt_read(self.path_test, False)

        return (self.index_train_data, self.index_test_data)

