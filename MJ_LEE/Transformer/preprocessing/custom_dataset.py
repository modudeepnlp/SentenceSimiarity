import re
import torch
import numpy as np
import config as config
import nltk
import os
import pickle
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
    def __init__(self, vocab_list, word_to_index):
        self.word_to_index = word_to_index
        self.vocab_list = list(word_to_index.keys())
        self.regex = config.regex
        self.label_to_index = {
            config.labels[0] : 0,  
            config.labels[1] : 1, 
            config.labels[2] : 2
            }

    def __PAD(self, index_array):
        while len(index_array) < config.max_seq_cut:
            index_array.append(1) # PAD_INDEX
        return index_array

    def __word_count_statistics(self, orgin_max_length):
        for i in range(max(orgin_max_length)): # 78
            print(i+1, " : ",  orgin_max_length.count(i+1))
      
    def __label_count_statistics(self, all_label):
        for label in config.labels: 
            print(label, " : ",  all_label.count(label))

    def __preprocessing(self, sentence):
        sentence = self.regex.sub('', sentence.lower()) 
        word_array = sentence.split(' ')
        
        if config.remove_outlier_seq:    
            if len(word_array) < config.min_seq_cut:
                return None, None, True
            if len(word_array) > config.max_seq_cut:
                return None, None, True

        index_array = [self.word_to_index[word] for word in word_array]
        index_array = self.__PAD(index_array)
        index_array = index_array[:config.max_seq]
        return index_array, len(index_array), False

    def __main_flow(self, path):
        all = []
        stat_label =[]
        index_data = []
        stat_length = []
        with open(path, "r") as f:
            for n, line in enumerate(f):
                if n == 0:
                    continue
                parts = line.strip().split("\t") 
                gold_label = parts[0] 
                if gold_label not in config.labels: 
                    continue
                index_label = self.label_to_index[gold_label]
                index_sent1, length1, is_continue1 = self.__preprocessing(parts[5])
                index_sent2, length2, is_continue2 = self.__preprocessing(parts[6])
                all.append(index_label)
                if is_continue1 or is_continue2:
                    continue
                index_data.append((index_label, index_sent1, index_sent2))
                stat_length.extend([length1, length2])
                stat_label.append(gold_label)
     
        if config.show_statistics:
            self.__word_count_statistics(stat_length)
            self.__label_count_statistics(stat_label)
        return index_data

    def __get_data(self, path):
        data = self.__main_flow(path)
        data = dummy_dataset(data)
        return data

    def get_data(self):
        index_train_data = self.__get_data(config.path_train)
        index_test_data = self.__get_data(config.path_test)
        index_dev_data = self.__get_data(config.path_dev)
        print("데이터 준비 완료")
        return (index_train_data, index_test_data, index_dev_data)