import sys
sys.path.append("..")

import os, random
from tqdm import tqdm, trange
import numpy as np
import pickle

import torch
import torch.utils.data
import torch.nn.functional as F

import global_data


"""
train dataset
"""
class XLNETDataSet(torch.utils.data.Dataset):
    def __init__(self, labels, sentence1s, sentence2s):
        self.labels = labels
        self.sentence1s = sentence1s
        self.sentence2s = sentence2s
    
    def __len__(self):
        assert len(self.sentence1s) == len(self.sentence2s)
        return len(self.sentence1s)
    
    def __getitem__(self, uid):
        label = self.labels[uid]
        sentence1 = self.sentence1s[uid]
        sentence2 = self.sentence2s[uid]
        sentence = []
        sentence.append(global_data.BOS_ID)
        sentence.extend(sentence1)
        sentence.append(global_data.SEP_ID)
        sentence.extend(sentence2)
        sentence.append(global_data.EOS_ID)
        return torch.tensor(label), torch.tensor(sentence)


"""
pretrain data iterator
"""
class XLNETPretrainDataSet(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, uid):
        feature = self.features[uid]
        feature = make_permute(feature,
                                reuse_len=128,
                                seq_len=256,
                                perm_size=128,
                                num_predict=43)

        return feature["input_k"], feature["seg_id"], feature["target"], feature["perm_mask"], feature["target_mapping"], feature["input_q"], feature["target_mask"]


"""
data loader 생성
"""
def build_data_loader(label, sentence1s, sentence2s, batch_size):
    dataset = XLNETDataSet(label, sentence1s, sentence2s)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)
    return loader


"""
data pretrain loader 생성
"""
def build_pretrain_loader(features, batch_size):
    dataset = XLNETPretrainDataSet(features)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pretrain_collate_fn)
    return loader


"""
data preprocessing
"""
def train_collate_fn(inputs):
    labels, sentences = list(zip(*inputs))

    sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=global_data.PAD_ID)

    batch = [
        torch.stack(labels, dim=0),
        sentences,
    ]
    return batch


"""
data preprocessing
"""
def pretrain_collate_fn(inputs):
    input_k, seg_id, target, perm_mask, target_mapping, input_q, target_mask = list(zip(*inputs))

    batch = [
        torch.stack(input_k, dim=0),
        torch.stack(seg_id, dim=0),
        torch.stack(target, dim=0),
        torch.stack(perm_mask, dim=0),
        torch.stack(target_mapping, dim=0),
        torch.stack(input_q, dim=0),
        torch.stack(target_mask, dim=0),
    ]
    return batch


def _split_a_and_b(data, sent_ids, begin_idx, tot_len, extend_target=False):
    """Split two segments from `data` starting from the index `begin_idx`."""

    data_len = data.shape[0]
    assert begin_idx + tot_len < data_len

    end_idx = begin_idx + 1
    cut_points = []
    while end_idx < data_len:
        if sent_ids[end_idx] != sent_ids[end_idx - 1]:
            if end_idx - begin_idx >= tot_len: break
            cut_points.append(end_idx)
        end_idx += 1
    
    a_begin = begin_idx
    if len(cut_points) == 0 or random.random() < 0.5: # notNext
        label = 0
        if len(cut_points) == 0:
            a_end = end_idx
        else:
            a_end = random.choice(cut_points)

        b_len = max(1, tot_len - (a_end - a_begin))
        b_begin = random.randint(0, data_len - 1 - b_len)
        b_end = b_begin + b_len
        while b_begin > 0 and sent_ids[b_begin - 1] == sent_ids[b_begin]:
            b_begin -= 1
        while b_end < data_len - 1 and sent_ids[b_end - 1] == sent_ids[b_end]:
            b_end += 1

        new_begin = a_end
    else: # isNext
        label = 1
        a_end = random.choice(cut_points)
        b_begin = a_end
        b_end = end_idx

        new_begin = b_end
    
    while a_end - a_begin + b_end - b_begin > tot_len:
        if a_end - a_begin > b_end - b_begin:
            a_end -= 1
        else:
            b_end -= 1
    
    ret = [data[a_begin: a_end], data[b_begin: b_end], label, new_begin]

    if extend_target:
        assert a_end< data_len and b_end < data_len
        a_target = data[a_begin + 1: a_end + 1]
        b_target = data[b_begin: b_end + 1]
        ret.extend([a_target, b_target])

    return ret


def _sample_mask(vocab, seg, mask_alpha, mask_beta, goal_num_predict, max_gram=5):
    """Sample `goal_num_predict` tokens for partial prediction.
    About `mask_beta` tokens are chosen in a context of `mask_alpha` tokens."""

    seg_len = len(seg)
    mask = np.array([False] * seg_len, dtype=np.bool)

    num_predict = 0

    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_gram + 1)
    pvals /= pvals.sum(keepdims=True)

    cur_len = 0
    while cur_len < seg_len:
        if num_predict >= goal_num_predict: break

        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - num_predict)
        ctx_size = (n * mask_alpha) // mask_beta
        l_ctx = np.random.choice(ctx_size)
        r_ctx = ctx_size - l_ctx

        beg = cur_len + l_ctx
        while beg < seg_len and not _is_start_piece(vocab.id_to_piece(seg[beg].item())):
            beg += 1
        if beg >= seg_len:
            break

        end = beg + 1
        cnt_ngram = 1
        while end < seg_len:
            if _is_start_piece(vocab.id_to_piece(seg[beg].item())):
                cnt_ngram += 1
                if cnt_ngram > n:
                    break
            end += 1
        if end >= seg_len:
            break

        mask[beg:end] = True
        num_predict += end - beg

        cur_len = end + r_ctx

    while num_predict < goal_num_predict:
        i = np.random.randint(seg_len)
        if not mask[i]:
            mask[i] = True
            num_predict += 1

    return mask


def _is_start_piece(piece):
    special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
    piece = ''.join(piece)
    if (piece.startswith("▁") or piece.startswith("<")
        or piece in special_pieces):
        return True
    else:
        return False


def _local_perm(inputs, targets, is_masked, perm_size, seq_len):
    """
    Sample a permutation of the factorization order, and create an
    attention mask accordingly.

    Args:
    inputs: int64 Tensor in shape [seq_len], input ids.
    targets: int64 Tensor in shape [seq_len], target ids.
    is_masked: bool Tensor in shape [seq_len]. True means being selected
      for partial prediction.
    perm_size: the length of longest permutation. Could be set to be reuse_len.
      Should not be larger than reuse_len or there will be data leaks.
    seq_len: int, sequence length.
    """

    index = torch.arange(seq_len, dtype=torch.int64)

    index = torch.reshape(index, [-1, perm_size]).t()
    index = index[torch.randperm(index.shape[0])]
    index = torch.reshape(index.t(), [-1])

    non_func_tokens = ~(torch.eq(inputs, global_data.SEP_ID) | torch.eq(inputs, global_data.CLS_ID))
    non_mask_tokens = (~is_masked.bool()) & non_func_tokens
    masked_or_func_tokens = ~non_mask_tokens

    smallest_index = -torch.ones([seq_len], dtype=torch.int64)

    rev_index = torch.where(non_mask_tokens, smallest_index, index)

    target_tokens = masked_or_func_tokens & non_func_tokens
    target_mask = target_tokens.type(torch.float32)

    self_rev_index = torch.where(target_tokens, rev_index, rev_index + 1)

    perm_mask = (self_rev_index[:, None] <= rev_index[None, :]) &  masked_or_func_tokens
    perm_mask = perm_mask.type(torch.float32)

    new_targets = torch.cat([inputs[0: 1], targets[: -1]], dim=0)

    inputs_k = inputs

    inputs_q = target_mask

    return perm_mask, new_targets, target_mask, inputs_k, inputs_q


def make_permute(in_feature, reuse_len, seq_len, perm_size, num_predict):
    inputs = torch.LongTensor(in_feature.get("input"))
    target = torch.LongTensor(in_feature.get("target"))
    is_masked = torch.ByteTensor(in_feature.get("is_masked"))
    feature = dict()

    non_reuse_len = seq_len - reuse_len
    assert perm_size <= reuse_len and perm_size <= non_reuse_len

    perm_mask_0, target_0, target_mask_0, input_k_0, input_q_0 = _local_perm(
        inputs[:reuse_len], # inp
        target[:reuse_len],
        is_masked[:reuse_len],
        perm_size,
        reuse_len)

    perm_mask_1, target_1, target_mask_1, input_k_1, input_q_1 = _local_perm(
        inputs[reuse_len:], # (senA, seq, senBm seq, cls)
        target[reuse_len:],
        is_masked[reuse_len:],
        perm_size,
        non_reuse_len)

    perm_mask_0 = torch.cat([perm_mask_0, torch.ones([reuse_len, non_reuse_len])], dim=1)
    perm_mask_1 = torch.cat([torch.zeros([non_reuse_len, reuse_len]), perm_mask_1], dim=1)

    perm_mask = torch.cat([perm_mask_0, perm_mask_1], dim=0)
    target = torch.cat([target_0, target_1], dim=0)
    target_mask = torch.cat([target_mask_0, target_mask_1], dim=0)
    input_k = torch.cat([input_k_0, input_k_1], dim=0)
    input_q = torch.cat([input_q_0, input_q_1], dim=0)

    if num_predict is not None:
        indices = torch.arange(seq_len, dtype=torch.int64)
        bool_target_mask = target_mask.bool()
        indices = indices[bool_target_mask]

        actual_num_predict = indices.shape[0]
        pad_len = num_predict - actual_num_predict

        assert seq_len >= actual_num_predict

        target_mapping = torch.eye(seq_len, dtype=torch.float32)[indices]
        paddings = torch.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
        target_mapping = torch.cat([target_mapping, paddings], dim=0)
        feature["target_mapping"] = torch.reshape(target_mapping,
                                                [num_predict, seq_len])
        # target = target[bool_target_mask]
        # paddings = torch.zeros([pad_len], dtype=target.dtype)
        # target = torch.cat([target, paddings], dim=0)
        feature["target"] = target.masked_fill_(~bool_target_mask, -1)

        target_mask = torch.cat(
            [torch.ones([actual_num_predict], dtype=torch.float32),
             torch.zeros([pad_len], dtype=torch.float32)],
            dim=0)
        feature["target_mask"] = torch.reshape(target_mask, [num_predict])
    else:
        feature["target"] = torch.reshape(target, [seq_len])
        feature["target_mask"] = torch.reshape(target_mask, [seq_len])

    feature["seg_id"] = torch.IntTensor(in_feature.get("seg_id"))
    feature["perm_mask"] = torch.reshape(perm_mask, [seq_len, seq_len])
    feature["input_k"] = torch.reshape(input_k, [seq_len])
    feature["input_q"] = torch.reshape(input_q, [seq_len])

    return feature


def _create_data(vocab, filename, seq_len, reuse_len, num_predict, mask_alpha, mask_beta, perm_size):
    features = []

    with open(filename, "r") as f:
        lines = f.readlines()
    input_data, sent_ids, sent_id = [], [], True

    count = 0
    for line in lines:
        count += 1
        if count % 100 == 0:
            print(f"vocab count: {count} / {len(lines)}", end="\r")
        cur_sent = vocab.encode_as_ids(line)
        input_data.extend(cur_sent)
        sent_ids.extend([sent_id] * len(cur_sent))
        sent_id = not sent_id
    
    data = np.array([input_data], dtype=np.int64)
    sent_ids = np.array([sent_ids], dtype=np.bool)

    assert reuse_len < seq_len - 3

    data_len = data.shape[1]
    sep_array = np.array([global_data.SEP_ID], dtype=np.int64)
    cls_array = np.array([global_data.CLS_ID], dtype=np.int64)

    count = 0
    i = 0
    print()
    while i + seq_len <= data_len:
        count += 1
        if count % 100 == 0:
            print(f"feature count: {count}", end="\r")
        inp = data[0, i: i + reuse_len]
        tgt = data[0, i + 1: i + reuse_len + 1]

        results = _split_a_and_b(
            data[0],
            sent_ids[0],
            begin_idx=i + reuse_len,
            tot_len=seq_len - reuse_len - 3,
            extend_target=True)

        (a_data, b_data, label, _, a_target, b_target) = tuple(results)

        if num_predict is None:
            num_predict_0 = num_predict_1 = None
        else:
            num_predict_1 = num_predict // 2
            num_predict_0 = num_predict - num_predict_1

        mask_0 = _sample_mask(vocab,
            inp,
            mask_alpha,
            mask_beta,
            goal_num_predict=num_predict_0)
        mask_1 = _sample_mask(vocab,
            np.concatenate([a_data, sep_array, b_data, sep_array, cls_array]),
            mask_alpha,
            mask_beta,
            goal_num_predict=num_predict_1)
        
        cat_data = np.concatenate([inp, a_data, sep_array, b_data, sep_array, cls_array])
        seg_id = ([0] * (reuse_len + a_data.shape[0]) + [0] + [1] * b_data.shape[0] + [1] + [2])

        assert cat_data.shape[0] == seq_len
        assert mask_0.shape[0] == seq_len // 2
        assert mask_1.shape[0] == seq_len // 2

        tgt = np.concatenate([tgt, a_target, b_target, cls_array, cls_array])
        assert tgt.shape[0] == seq_len

        is_masked = np.concatenate([mask_0, mask_1], 0)
        if num_predict is not None:
            assert np.sum(is_masked) == num_predict

        feature = {
            "input": cat_data,
            "is_masked": is_masked,
            "target": tgt,
            "seg_id": seg_id,
            "label": [label],
        }
        features.append(feature)

        i += reuse_len
    
    return features


"""
pretrain data create and dump
"""
def demp_pretrain(vocab_file, file):
    in_file = f"../data/corpus.book.middle.txt"

    vocab = global_data.load_vocab(vocab_file)

    features = _create_data(vocab=vocab,
                filename=in_file,
                seq_len=256,
                reuse_len=128,
                num_predict=43,
                mask_alpha=6,
                mask_beta=1,
                perm_size=128)
    
    with open(file, 'wb') as f:
        pickle.dump((features), f)


"""
pretrain data load
"""
def load_pretrain(file):
    with open(file, 'rb') as f:
        features = pickle.load(f)
    return features


if __name__ == "__main__":
    demp_pretrain("../data/m_snli_8000.model", "../data/pretrain_xlnet_8000_0.pkl")
    # demp_pretrain("../data/m_snli_16000.model", "../data/pretrain_xlnet_16000_0.pkl")
    pass

