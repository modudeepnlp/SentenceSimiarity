import re
import torch
import numpy as np
import config as config
import nltk
import collections
import os
import pickle

from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize 
from torch.utils import data


class dummy_dataset(data.Dataset):
    def __init__(self, custom_dataset):        
        self.dataset = custom_dataset
        self.len = len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.len

class Custom_dataset():
    def __init__(self):
        path = "../../data/snli_1.0/"
        self.path_train = path + "snli_1.0_train.txt"
        self.path_test = path + "snli_1.0_test.txt"
        self.path_dev = path + "snli_1.0_dev.txt"

        self.train_mode = 0
        self.test_mode = 1 
        self.dev_mode = 2

        self.stop_words = stopwords.words('english')
        self.UNK = "[UNK]"
        self.PAD = "[PAD]"

        self.labels = ["neutral", "contradiction", "entailment"] 
        self.label_to_index = {
            self.labels[0] : 0,  
            self.labels[1] : 1, 
            self.labels[2] : 2
            }

    def __stop_words(self, word_array):
        if config.use_stop_word:
            result = [] 
            for w in word_array: 
                if w not in self.stop_words: 
                    result.append(w) 
            return result
        return word_array

    def __regular_expresion(self, sentence):
        clear = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sentence)
        return clear.lower()

    def __get_all_words_array(self, sentences):
        one_line_sentences = ' '.join(sentence for sentence in sentences)
        all_words_array = one_line_sentences.split(' ')
        return all_words_array

    def __indexing_vocab(self, all_words_array):
        print("--indexing start--")
        self.no_duplication_words = [self.UNK, self.PAD] # PAD Index = 1
        self.no_duplication_words.extend(list(sorted(set(all_words_array))))
        self.no_duplication_words = self.__stop_words(self.no_duplication_words)
        self.word_to_index = { word:i for i,word in enumerate(self.no_duplication_words) }
        self.index_to_word = { i:word for i,word in enumerate(self.no_duplication_words) }
        self.vocab_size = len(self.word_to_index)

    def __find_missing_word(self, word):
        if [word] not in self.no_duplication_words: # 찾는 형태가 배열이여야지 찾음
            return self.UNK
        else:
            return word

    def __UNK(self, str_train_data):
        print("--UNK start--")
        result = []
        length = len(str_train_data)
        for n, (gold_label, word_array1, word_array2) in enumerate(str_train_data):
            if n % 3000 == 0:
                print((n*100)/length,"%")
            new_word_array1 = [self.__find_missing_word(word) for word in word_array1]
            new_word_array2 = [self.__find_missing_word(word) for word in word_array2]   
            result.append((gold_label, new_word_array1, new_word_array2))
        print("100%")
        return result

    def __remove_low_frequency_words(self, all_words_array):
        if config.use_remove_freq:
            unique = collections.Counter(all_words_array)  
            word_freq = unique.most_common() 
            length = len(word_freq)
            
            print("remove low frequency words")
            result = []
            low_freq_word = []
            for n, pair in enumerate(word_freq):
                word = pair[0]
                freq = pair[1]
                if freq < config.low_freq:
                    low_freq_word.append(word)
                    continue
                result.append(word)
            print("low_freq_word_count : ", len(low_freq_word))
            print("high_freq_word_count : ", len(result))

            string = ' '.join([str(i) for i in low_freq_word])
            f = open(config.low_freq_words_file , 'w') # 확인용
            f.write(string)
            f.close()
            print("낮은 빈도 단어 리스트가 저장 되었다.")
            return result
        return all_words_array

    def __PAD(self, index_array):
        while len(index_array) < config.max_seq:
            index_array.append(1) # PAD_INDEX
        return index_array

    def __Max_length(self, index_array):
        if len(index_array) > config.max_seq:
            index_array = index_array[:config.max_seq]
        return index_array

    def __word_count_statistics(self, orgin_max_length):
        for i in range(max(orgin_max_length)): # 78
            print(i+1, " : ",  orgin_max_length.count(i+1))
      
    def __label_count_statistics(self, all_label):
        for label in self.labels: 
            print(label, " : ",  all_label.count(label))

    def __save_words(self):
        with open(config.vocab_list, 'wb') as f:
            pickle.dump(self.no_duplication_words, f)
        print("vocab_list 가 저장되었다.")
        with open(config.word_to_index, 'wb') as f:
            pickle.dump(self.word_to_index, f)
        print("word_to_index 가 저장되었다.")
        with open(config.index_to_word, 'wb') as f:
            pickle.dump(self.index_to_word, f)
        print("index_to_word 가 저장되었다.")
        with open(config.vocab_size, 'wb') as f:
            pickle.dump(self.vocab_size, f)
        print("vocab_size 가 저장되었다.")

    def __load_words(self):
        with open(config.vocab_list, 'rb') as f:
            self.no_duplication_words = pickle.load(f) 
        print("vocab_list 를 불러왔다.")
        with open(config.word_to_index, 'rb') as f:
            self.word_to_index = pickle.load(f) 
        print("word_to_index 를 불러왔다.")
        with open(config.index_to_word, 'rb') as f:
            self.index_to_word = pickle.load(f) 
        print("index_to_word 를 불러왔다.")
        with open(config.vocab_size, 'rb') as f:
            self.vocab_size = pickle.load(f) 
        print("vocab_size 를 불러왔다.")

    def __main_flow(self, path, mode):
        if mode ==  self.train_mode:
            print("--train data 전처리를 시작합니다--")
        elif mode ==  self.test_mode:
            print("--test data 전처리를 시작합니다--")
        else:
            print("--dev data 전처리를 시작합니다--")

        #################################################################### 문자열 전처리 (regex, stop word)
        data = []
        all_label =[]
        all_sentences = []
        with open(path, "r") as f:
            for n, line in enumerate(f):
                if n == 0:
                    continue
                parts = line.strip().split("\t") # 한 줄당 여러 탭으로 파트가 나뉘어 져 있음
                gold_label = parts[0] 
                sentence1 = self.__regular_expresion(parts[5]) 
                sentence2 = self.__regular_expresion(parts[6]) 
                word_array1 = sentence1.split(' ')
                word_array2 = sentence2.split(' ')
                word_array1 = self.__stop_words(word_array1)
                word_array2 = self.__stop_words(word_array2)
                all_label.append(gold_label)
                all_sentences.append(sentence1)
                all_sentences.append(sentence2)

                data.append((gold_label, word_array1, word_array2))

        print("--Done sentence to word array--")
        self.__label_count_statistics(all_label)

        #################################################################### index 생성, UNK 전환
        if mode ==  self.train_mode:
            all_words_array = self.__get_all_words_array(all_sentences)
            all_words_array = self.__remove_low_frequency_words(all_words_array)
            self.__indexing_vocab(all_words_array) 
            self.__save_words()
        else:
            self.__load_words()
            data = self.__UNK(data)
        # data = self.__UNK(data)

        #################################################################### 문자열 전처리 (regex, stop word)
        print("--start indexing, UNK, PAD, Max_Length--")
        index_data = []
        data2 = []
        origin_max_length = []
        for n, data_tuple in enumerate(data):
            (gold_label, word_array1, word_array2) = data_tuple
            if gold_label not in self.labels: # 레이블 없는 것 예외처리
                continue
            # if len(word_array1) < config.min_seq_cut:
            #     continue
            # if len(word_array2) < config.min_seq_cut:
            #     continue
            # if len(word_array1) > config.max_seq_cut:
            #     continue
            # if len(word_array2) > config.max_seq_cut:
            #     continue
            
            index_label = self.label_to_index[gold_label]
            index_sent1 = [self.word_to_index[word] for word in word_array1]
            index_sent2 = [self.word_to_index[word] for word in word_array2]
            origin_max_length.append(len(index_sent1))
            origin_max_length.append(len(index_sent2))

            data2.append(index_label)
            # max_len 모자란 길이 PAD 입력
            index_sent1 = self.__PAD(index_sent1)
            index_sent2 = self.__PAD(index_sent2)

            # max_len 넘는 배열 제거
            index_sent1 = self.__Max_length(index_sent1)
            index_sent2 = self.__Max_length(index_sent2)

            index_data.append((index_label, index_sent1, index_sent2))
     
        self.__word_count_statistics(origin_max_length)
        self.__label_count_statistics(all_label)

        print("--전처리가 끝났습니다--")
        return index_data

    def __load_or_preprocessing(self, mode):
        if mode ==  self.train_mode:
            path = self.path_train
            file_name = config.preprocessed_train_file
        elif mode == self.test_mode:
            path = self.path_test
            file_name = config.preprocessed_test_file
        else:
            path = self.path_dev
            file_name = config.preprocessed_dev_file

        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                data = pickle.load(f) # 전처리된 데이터
        else:
            data = self.__main_flow(path, mode)
            with open(file_name, 'wb') as f:
                pickle.dump(data, f)
        return data

    def get_data(self):
        index_train_data = self.__load_or_preprocessing(self.train_mode)
        index_test_data = self.__load_or_preprocessing(self.test_mode)
        index_dev_data = self.__load_or_preprocessing(self.dev_mode)

        index_train_data = dummy_dataset(index_train_data)
        index_test_data = dummy_dataset(index_test_data)
        index_dev_data = dummy_dataset(index_dev_data)

        with open(config.vocab_size, 'rb') as f:
            self.vocab_size = pickle.load(f) 

        return (index_train_data, index_test_data, index_dev_data)

