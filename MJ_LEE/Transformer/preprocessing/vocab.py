import config as config
import re
import collections
import os
import json

class create_vocab():
    def __init__(self, mode):
        self.regex = config.regex
        if mode == config.all_mode:
            self.path_list = [config.path_train, config.path_test, config.path_dev]
        elif mode == config.train_mode:
            self.path_list = [config.path_train]
        elif mode == config.test_mode:
            self.path_list = [config.path_test]
        else:
            self.path_list = [config.path_dev]

    def __save_words(self):
        with open(config.vocab_list, 'w') as f:
            json.dump(self.__vocab_list, f)

    def __load_words(self):
        with open(config.vocab_list, 'r') as f:    
            self.__vocab_list = json.load(f)

    def __main_flow(self):
        all_sentences = []
        for path in self.path_list:
            with open(path, "r") as f:
                for n, line in enumerate(f):
                    if n == 0: # 첫줄은 무의미한 문장
                        continue
                    parts = line.strip().split("\t") # 한 줄당 여러 탭으로 파트가 나뉘어 져 있음
                    gold_label = parts[0] 
                    sentence1 = self.regex.sub('', parts[5])  
                    sentence2 = self.regex.sub('', parts[6])
                    all_sentences.append(sentence1.lower())
                    all_sentences.append(sentence2.lower())

        one_line = ' '.join(sentence for sentence in all_sentences)
        all_words = one_line.split(' ')
        self.__vocab_list = [config.UNK, config.PAD] 
        self.__vocab_list.extend(list(sorted(set(all_words))))
        self.__word_to_index = { word:i for i,word in enumerate(self.__vocab_list) }
        self.__index_to_word = { i:word for i,word in enumerate(self.__vocab_list) }

    def get_data(self):
        self.__main_flow()
        print("vocab_size : ", len(self.__vocab_list))
        return self.__vocab_list, self.__word_to_index

