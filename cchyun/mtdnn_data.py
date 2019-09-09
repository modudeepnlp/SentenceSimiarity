import os, sys, random
import pandas as pd
import json, yaml
from tqdm import tqdm, trange
from shutil import copyfile
from enum import Enum, IntEnum

import torch
import torch.nn.functional as F

import data
import tokenizer


class TaskType(IntEnum):
 Classification = 1
 Regression = 2
 Ranking = 3


class DataFormat(IntEnum):
    PremiseOnly = 1
    PremiseAndOneHypothesis = 2
    PremiseAndMultiHypothesis = 3


class EncoderModelType(IntEnum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3


class Metric(Enum):
    ACC = 0
    F1 = 1
    MCC = 2
    Pearson = 3
    Spearman = 4
    AUC = 5


class Vocabulary(object):
    INIT_LEN = 4
    def __init__(self, neat=False):
        self.neat = neat
        if not neat:
            self.tok2ind = {PAD: PAD_ID, UNK: UNK_ID, STA: STA_ID, END: END_ID}
            self.ind2tok = {PAD_ID: PAD, UNK_ID: UNK, STA_ID: STA, END_ID:END}
        else:
            self.tok2ind = {}
            self.ind2tok = {}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return key in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, -1) if self.neat else self.ind2tok.get(key, UNK)
        if type(key) == str:
            return self.tok2ind.get(key, None) if self.neat else self.tok2ind.get(key,self.tok2ind.get(UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def get_vocab_list(self, with_order=True):
        if with_order:
            words = [self[k] for k in range(0, len(self))]
        else:
            words = [k for k in self.tok2ind.keys()
                      if k not in {PAD, UNK, STA, END}]
        return words

    def toidx(self, tokens):
        return [self[tok] for tok in tokens]

    def copy(self):
        """Deep copy
        """
        new_vocab = Vocabulary(self.neat)
        for w in self:
            new_vocab.add(w)
        return new_vocab

    def build(words, neat=False):
        vocab = Vocabulary(neat)
        for w in words: vocab.add(w)
        return vocab


class TaskDefs:
    def __init__(self, task_def_path):
        self._task_def_dic = yaml.safe_load(open(task_def_path))
        global_map = {}
        n_class_map = {}
        data_type_map = {}
        task_type_map = {}
        metric_meta_map = {}
        enable_san_map = {}
        dropout_p_map = {}
        encoderType_map = {}
        uniq_encoderType = set()
        for task, task_def in self._task_def_dic.items():
            assert "_" not in task, "task name should not contain '_', current task name: %s" % task
            n_class_map[task] = task_def["n_class"]
            data_format = DataFormat[task_def["data_format"]]
            data_type_map[task] = data_format
            task_type_map[task] = TaskType[task_def["task_type"]]
            metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def["metric_meta"])
            enable_san_map[task] = task_def["enable_san"]
            uniq_encoderType.add(EncoderModelType[task_def["encoder_type"]])
            if "labels" in task_def:
                labels = task_def["labels"]
                label_mapper = Vocabulary(True)
                for label in labels:
                    label_mapper.add(label)
                global_map[task] = label_mapper
            if "dropout_p" in task_def:
                dropout_p_map[task] = task_def["dropout_p"]

        assert len(uniq_encoderType) == 1, 'The shared encoder has to be the same.'
        self.global_map = global_map
        self.n_class_map = n_class_map
        self.data_type_map = data_type_map
        self.task_type_map = task_type_map
        self.metric_meta_map = metric_meta_map
        self.enable_san_map = enable_san_map
        self.dropout_p_map = dropout_p_map
        self.encoderType = uniq_encoderType.pop()


UNK_ID=100
BOS_ID=101

class BatchGen:
    def __init__(self, data, batch_size=32, gpu=True, device="cpu", is_train=True,
                 maxlen=128, dropout_w=0.005,
                 do_batch=True, weighted_on=False,
                 task_id=0,
                 pairwise=False,
                 task=None,
                 task_type=TaskType.Classification,
                 data_type=DataFormat.PremiseOnly,
                 soft_label=False,
                 encoder_type=EncoderModelType.BERT):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.is_train = is_train
        self.gpu = gpu
        self.device = device
        self.weighted_on = weighted_on
        self.data = data
        self.task_id = task_id
        self.pairwise = pairwise
        self.task = task
        self.pairwise_size = 1
        self.data_type = data_type
        self.task_type=task_type
        self.encoder_type = encoder_type
        # soft label used for knowledge distillation
        self.soft_label_on = soft_label
        if do_batch:
            if is_train:
                indices = list(range(len(self.data)))
                random.shuffle(indices)
                data = [self.data[i] for i in indices]
            self.data = BatchGen.make_baches(data, batch_size)
        self.offset = 0
        self.dropout_w = dropout_w

    @staticmethod
    def make_baches(data, batch_size=32):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    @staticmethod
    def load(path, is_train=True, maxlen=128, factor=1.0, pairwise=False):
        with open(path, 'r', encoding='utf-8') as reader:
            data = []
            cnt = 0
            for line in reader:
                sample = json.loads(line)
                sample['factor'] = factor
                cnt += 1
                if is_train:
                    if pairwise and (len(sample['token_id'][0]) > maxlen or len(sample['token_id'][1]) > maxlen):
                        continue
                    if (not pairwise) and (len(sample['token_id']) > maxlen):
                        continue
                data.append(sample)
            print('Loaded {} samples out of {}'.format(len(data), path))
            return data

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
        self.offset = 0

    def __random_select__(self, arr):
        if self.dropout_w > 0:
            return [UNK_ID if random.uniform(0, 1) < self.dropout_w else e for e in arr]
        else: return arr

    def __len__(self):
        return len(self.data)

    def patch(self, v):
        # v = v.cuda(non_blocking=True)
        v = v.to(self.device)
        return v

    @staticmethod
    def todevice(v, device):
        v = v.to(device)
        return v

    def rebacth(self, batch):
        newbatch = []
        for sample in batch:
            size = len(sample['token_id'])
            self.pairwise_size = size
            assert size == len(sample['type_id'])
            for idx in range(0, size):
                token_id = sample['token_id'][idx]
                type_id = sample['type_id'][idx]
                uid = sample['ruid'][idx]
                olab = sample['olabel'][idx]
                newbatch.append({'uid': uid, 'token_id': token_id, 'type_id': type_id, 'label':sample['label'], 'true_label': olab})
        return newbatch

    def __if_pair__(self, data_type):
        return data_type in [DataFormat.PremiseAndOneHypothesis, DataFormat.PremiseAndMultiHypothesis]

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            if self.pairwise:
                batch = self.rebacth(batch)
            batch_size = len(batch)
            batch_dict = {}
            tok_len = max(len(x['token_id']) for x in batch)
            hypothesis_len = max(len(x['type_id']) - sum(x['type_id']) for x in batch)
            if self.encoder_type == EncoderModelType.ROBERTA:
                token_ids = torch.LongTensor(batch_size, tok_len).fill_(1)
                type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
                masks = torch.LongTensor(batch_size, tok_len).fill_(0)
            else:
                token_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
                type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
                masks = torch.LongTensor(batch_size, tok_len).fill_(0)
            if self.__if_pair__(self.data_type):
                premise_masks = torch.ByteTensor(batch_size, tok_len).fill_(1)
                hypothesis_masks = torch.ByteTensor(batch_size, hypothesis_len).fill_(1)

            for i, sample in enumerate(batch):
                select_len = min(len(sample['token_id']), tok_len)
                tok = sample['token_id']
                if self.is_train:
                    tok = self.__random_select__(tok)
                token_ids[i, :select_len] = torch.LongTensor(tok[:select_len])
                type_ids[i, :select_len] = torch.LongTensor(sample['type_id'][:select_len])
                masks[i, :select_len] = torch.LongTensor([1] * select_len)
                if self.__if_pair__(self.data_type):
                    hlen = len(sample['type_id']) - sum(sample['type_id'])
                    hypothesis_masks[i, :hlen] = torch.LongTensor([0] * hlen)
                    for j in range(hlen, select_len):
                        premise_masks[i, j] = 0
            if self.__if_pair__(self.data_type):
                batch_info = {
                    'token_id': 0,
                    'segment_id': 1,
                    'mask': 2,
                    'premise_mask': 3,
                    'hypothesis_mask': 4
                    }
                batch_data = [token_ids, type_ids, masks, premise_masks, hypothesis_masks]
                current_idx = 5
                valid_input_len = 5
            else:
                batch_info = {
                    'token_id': 0,
                    'segment_id': 1,
                    'mask': 2
                    }
                batch_data = [token_ids, type_ids, masks]
                current_idx = 3
                valid_input_len = 3

            if self.is_train:
                labels = [sample['label'] for sample in batch]
                if self.task_type == TaskType.Regression:
                    batch_data.append(torch.FloatTensor(labels))
                else:
                    batch_data.append(torch.LongTensor(labels))
                batch_info['label'] = current_idx
                current_idx += 1
                # soft label generated by ensemble models for knowledge distillation
                if self.soft_label_on and (batch[0].get('softlabel', None) is not None):
                    sortlabels = [sample['softlabel'] for sample in batch]
                    sortlabels = torch.FloatTensor(sortlabels)
                    batch_info['soft_label'] = self.patch(sortlabels.pin_memory()) if self.gpu else sortlabels

            if self.gpu:
                for i, item in enumerate(batch_data):
                    batch_data[i] = self.patch(item.pin_memory())

            # meta 
            batch_info['uids'] = [sample['uid'] for sample in batch]
            batch_info['task'] = self.task
            batch_info['task_id'] = self.task_id
            batch_info['input_len'] = valid_input_len
            batch_info['pairwise'] = self.pairwise
            batch_info['pairwise_size'] = self.pairwise_size
            batch_info['task_type'] = self.task_type
            if not self.is_train:
                labels = [sample['label'] for sample in batch]
                batch_info['label'] = labels
                if self.pairwise:
                    batch_info['true_label'] = [sample['true_label'] for sample in batch]
            self.offset += 1
            yield batch_info, batch_data


def build_data_loader(config, vocab, task_defs, mode, datasets):
    data_list = []
    tasks = {}
    tasks_class = {}
    dropout_list = []
    for dataset in datasets:
        filepath = os.path.join("data/canonical_data/cchyun_data", '{}_{}.json'.format(dataset, mode))
        if not os.path.isfile(filepath):
            continue

        prefix = dataset.split('_')[0]
        if prefix in tasks: continue
        assert prefix in task_defs.n_class_map
        assert prefix in task_defs.data_type_map
        # DataFormat [PremiseOnly, PremiseAndOneHypothesis, PremiseAndMultiHypothesis]
        data_type = task_defs.data_type_map[prefix]
        # count of classes
        nclass = task_defs.n_class_map[prefix]
        task_id = len(tasks)
        # TaskType [Classification, Regression, Ranking]
        task_type = task_defs.task_type_map[prefix]
        pw_task = True if task_type == TaskType.Ranking else False

        if prefix not in tasks:
            tasks[prefix] = task_id # len(tasks)

        if (nclass not in tasks_class):
            tasks_class[nclass] = len(tasks_class)

        dropout = task_defs.dropout_p_map.get(prefix, config.dropout)
        dropout_list.append(dropout)

        taskdata = BatchGen(
                        BatchGen.load(filepath, True, pairwise=pw_task, maxlen=config.n_enc_seq),
                        batch_size=config.n_batch,
                        dropout_w=config.dropout,
                        gpu=True,
                        device=config.device,
                        maxlen=config.n_enc_seq,
                        task_id=task_id,
                        pairwise=pw_task,
                        task=prefix,
                        data_type=data_type,
                        task_type=task_type)
        data_list.append(taskdata)
    return data_list


label_dict = {
        "neutral": 0., "entailment": 1., "contradiction": 2.,
        "entails": 1.
    }

def build_canonical_line(vocab, cols):
    if cols[1].strip() == "not_entailment":
        return None
    label = cols[1].strip()
    if label in label_dict:
        label = label_dict[label]
    else:
        label = float(label)

    data = {"uid": cols[0].strip(), "label": label}
    if len(cols) == 3:
        token_id = []
        type_id = []
        seq1s = tokenizer.tokenize(cols[2].lower())
        token_id.append(vocab["<cls>"])
        for seq in seq1s:
            if seq in vocab:
                token_id.append(vocab[seq])
            else:
                token_id.append(vocab["<unk>"])
        token_id.append(vocab["<sep>"])
        type_id.extend([0] * (len(seq1s) + 2))
        data["token_id"] = token_id
        data["type_id"] = type_id
    elif len(cols) == 4:
        token_id = []
        type_id = []
        seq1s = tokenizer.tokenize(cols[2].lower())
        seq2s = tokenizer.tokenize(cols[3].lower())
        token_id.append(vocab["<cls>"])
        for seq in seq1s:
            if seq in vocab:
                token_id.append(vocab[seq])
            else:
                token_id.append(vocab["<unk>"])
        token_id.append(vocab["<sep>"])
        for seq in seq2s:
            if seq in vocab:
                token_id.append(vocab[seq])
            else:
                token_id.append(vocab["<unk>"])
        token_id.append(vocab["<sep>"])
        type_id.extend([0] * (len(seq1s) + 2))
        type_id.extend([1] * (len(seq2s) + 1))
        data["token_id"] = token_id
        data["type_id"] = type_id
    else:
        print(cols)
        return None
    return data


def build_canonical_one(vocab, srcfile, dstfile):
    with open(srcfile, "r") as src:
        with open(dstfile, "w") as dst:
            for line in src:
                cols = line.split("\t")
                if len(cols) == 3 or len(cols) == 4:
                    data = build_canonical_line(vocab, cols)
                    if data is not None:
                        dst.write(json.dumps(data))
                        dst.write("\n")


def build_canonical_data(vocab):
    dirname = "data/canonical_data"
    filenames = os.listdir(dirname)
    for filename in tqdm(filenames):
        srcfile = os.path.join(dirname, filename)
        if os.path.isfile(srcfile) and srcfile.endswith(".tsv"):
            # print(filename)
            basename = os.path.basename(srcfile)
            dstfile = (dirname + "/cchyun_data/" + basename).replace(".tsv", ".json")
            build_canonical_one(vocab, srcfile, dstfile)


if __name__ == "__main__":
    vocab, train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2, max_sentence1, max_sentence2, max_sentence_all = data.load_data("data/snli_data.pkl")
    build_canonical_data(vocab)

