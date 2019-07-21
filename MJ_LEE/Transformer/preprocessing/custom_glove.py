import config as config
import torch
import os
import numpy as np
import json

class custom_glove():
    def __init__(self):
        self.glove_json = {}
    def __get_glove_json(self):
        if os.path.exists(config.glove_json):
            with open(config.glove_json, 'r') as f:    
                self.glove_json = json.load(f)
        else:
            print("Read glove.840B.300d.txt ...")
            with open(config.glove, encoding="utf-8") as f:
                for line in f:
                    values = line.split(' ')
                    word = values[0]
                    coefs = values[1:] 
                    self.glove_json[word] = coefs
            with open(config.glove_json, 'w') as f:
                json.dump(self.glove_json, f)

    def get_data(self, vocab_list):
        if os.path.exists(config.glove_npy):
            embedding = np.load(config.glove_npy)
        else:
            self.__get_glove_json()
            print("pretrain embedding 생성중...")
            embedding = np.zeros((len(vocab_list), 300))
            for index, word in enumerate(vocab_list): # vocab에 있는 순서 그대로
                vector = self.glove_json.get(word)
                if vector is not None:
                    embedding[index] = np.asarray(vector, dtype='float32') 
            np.save(config.glove_npy, embedding)
        embedding = torch.FloatTensor(embedding)
        return embedding





    