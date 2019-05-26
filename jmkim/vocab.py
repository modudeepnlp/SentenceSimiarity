import itertools
import pickle
import pandas as pd
import gluonnlp as nlp

from pathlib import Path
from mecab import MeCab

#train path
train_path = Path.cwd() / 'data_in' / 'snli_1.0_train.txt'
# train data를 tab으로 구별 document, label 컬럼으로 불러옴
tr = pd.read_csv(train_path, sep='\t').loc[:, ['sentence1', 'sentence2']]
# Mecab 정의
tokenizer = MeCab()
# document 열의 데이터를 Mecab의 형태소로 나눈 것들을 list로 변환
tokenized = tr['sentence1'].apply(lambda elm: tokenizer.morphs(str(elm))).tolist()
tokenized += tr['sentence2'].apply(lambda elm: tokenizer.morphs(str(elm))).tolist()
# tokenized 에서 각 단어의 count 저장
counter = nlp.data.count_tokens(itertools.chain.from_iterable(tokenized))

# counter에서 최소 10번 이상 나온것들을 vocab에 저장
vocab = nlp.Vocab(counter=counter, min_freq=10, bos_token=None, eos_token=None)

nlp.embedding.list_sources()
# wiki.ko 데이터를 fasttext로 벡터화 한 임베딩 가져오기
embedding = nlp.embedding.create('fasttext', source='wiki.ko')

# 만든 vocab에 벡터 적용
vocab.set_embedding(embedding)

# vocab.pkl 저장
with open(Path.cwd() / 'data_in' / 'vocab.pkl', mode='wb') as io:
    pickle.dump(vocab, io)
