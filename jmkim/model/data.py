import pandas as pd
import torch
from torch.utils.data import Dataset
from mecab import MeCab
from gluonnlp.data import PadSequence
from gluonnlp import Vocab
from typing import Tuple

class Corpus(Dataset):
    def __init__(self, filepath: str, vocab: Vocab, tokenizer: MeCab, padder: PadSequence):
        self._corpus = pd.read_csv(filepath, sep='\t').iloc[:, [0, 1, 2]]
        self._vocab = vocab
        self._toknizer = tokenizer
        self._padder = padder

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx):
        label_dict = {"neutral": 0, "entailment": 1, "contradiction": 2, "-" : 3}

        try:
            sen = self._toknizer.morphs(self._corpus.iloc[idx][0])
        except:
            sen = self._toknizer.morphs('')
        sen2indices = torch.tensor(self._padder([self._vocab.token_to_idx[token] for token in sen]), dtype=torch.long)

        try:
            sen2 = self._toknizer.morphs(self._corpus.iloc[idx][1])
        except:
            sen2 = self._toknizer.morphs('')
        sen22indices = torch.tensor(self._padder([self._vocab.token_to_idx[token] for token in sen2]), dtype=torch.long)

        gold_label = torch.tensor(label_dict[self._corpus.iloc[idx][2]], dtype=torch.long)
        return gold_label, sen2indices, sen22indices
