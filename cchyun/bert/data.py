import sys
sys.path.append("..")

import os
from tqdm import tqdm, trange
import numpy as np
import pickle

import collections
from tempfile import TemporaryDirectory
from pathlib import Path
import shelve
from random import random, randrange, randint, shuffle, choice
import json
import logging

import torch
import torch.utils.data
import torch.nn.functional as F

import config as cfg
import global_data


"""
train dataset
"""
class BERTDataSet(torch.utils.data.Dataset):
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
        sentence = []
        sentence.append(global_data.CLS_ID)
        sentence.extend(sentence1)
        sentence.append(global_data.SEP_ID)
        sentence.extend(sentence2)
        sentence.append(global_data.SEP_ID)
        segment = []
        segment.extend([0] * (len(sentence1) + 2))
        segment.extend([1] * (len(sentence2) + 1))
        return torch.tensor(label), torch.tensor(sentence), torch.tensor(segment)


"""
data loader 생성
"""
def build_data_loader(label, sentence1s, sentence2s, batch_size):
    dataset = BERTDataSet(label, sentence1s, sentence2s)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader


"""
data preprocessing
"""
def collate_fn(inputs):
    labels, sentences, segment = list(zip(*inputs))

    sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=global_data.PAD_ID)
    segment = torch.nn.utils.rnn.pad_sequence(segment, batch_first=True, padding_value=global_data.PAD_ID)

    batch = [
        torch.stack(labels, dim=0),
        sentences,
        segment,
    ]
    return batch


########################################################################
# start of pregenerate_training_data.py
########################################################################
class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (whole_word_mask and len(cand_indices) >= 1 and not token.startswith(u"\u2581")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels


def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document = doc_database[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = randrange(1, len(current_chunk))

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []

                # Random next
                if len(current_chunk) == 1 or random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # Sample a random document, with longer docs being sampled more frequently
                    random_document = doc_database.sample_doc(current_idx=doc_idx, sentence_weighted=True)

                    random_start = randrange(0, len(random_document))
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                if 0 < len(tokens_a) and 0 < len(tokens_b):
                    # assert len(tokens_a) >= 1
                    # assert len(tokens_b) >= 1

                    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                    # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                    # They are 1 for the B tokens and the final [SEP]
                    segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                        tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list)

                    instance = {
                        "tokens": tokens,
                        "segment_ids": segment_ids,
                        "is_random_next": is_random_next,
                        "masked_lm_positions": masked_lm_positions,
                        "masked_lm_labels": masked_lm_labels}
                    instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def create_training_file(docs, vocab_list, args):
    epoch_filename = Path(f"{args.save_filename}_data.json")
    num_instances = 0
    with epoch_filename.open('w') as epoch_file:
        for doc_idx in trange(len(docs), desc="Document"):
            doc_instances = create_instances_from_document(
                docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list)
            doc_instances = [json.dumps(instance) for instance in doc_instances]
            for instance in doc_instances:
                epoch_file.write(instance + '\n')
                num_instances += 1
    metrics_file = Path(f"{args.save_filename}_metrics.json")
    with metrics_file.open('w') as metrics_file:
        metrics = {
            "num_training_examples": num_instances,
            "max_seq_len": args.max_seq_len
        }
        metrics_file.write(json.dumps(metrics))


def convert_example_to_features(example, vocab, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    # input_ids = convert_tokens_to_ids(vocab, tokens)
    input_ids = [vocab.piece_to_id(token) for token in tokens]
    # masked_label_ids = convert_tokens_to_ids(vocab, masked_lm_labels)
    masked_label_ids = [vocab.piece_to_id(token) for token in masked_lm_labels]

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    return cfg.Config({
        "input_ids": input_array,
        "segment_ids": segment_array,
        "lm_label_ids": lm_label_array,
        "is_next": is_random_next
    })


class PregeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, path, vocab, reduce_memory=False):
        self.vocab = vocab
        data_file = Path(f"{path}_data.json")
        metrics_file = Path(f"{path}_metrics.json")
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir/'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
        logging.info(f"Loading training examples for")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, vocab, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)))
########################################################################
# end of pregenerate_training_data.py
########################################################################


"""
pretrain data create and dump
"""
def demp_pretrain(vocab_file, file):
    args = cfg.Config({
        "reduce_memory": False,
        "train_corpus": Path("../data/corpus.book.large.txt"),
        "output_dir": Path("data"),
        "max_seq_len": 256,
        "short_seq_prob": 0.1,
        "masked_lm_prob": 0.15,
        "max_predictions_per_seq": 20,
        "do_whole_word_mask": True,
        "save_filename": file,
    })

    vocab = global_data.load_vocab(vocab_file)

    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        with args.train_corpus.open() as f:
            doc = []
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                line = line.strip()
                if line == "":
                    docs.add_document(doc)
                    doc = []
                else:
                    tokens = vocab.encode_as_pieces(line.lower())
                    doc.append(tokens)
            if doc:
                docs.add_document(doc)  # If the last doc didn't end on a newline, make sure it still gets added
        if len(docs) <= 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")
    
    vocab_list = []
    for id in range(vocab.get_piece_size()):
        if not vocab.is_unknown(id):
            vocab_list.append(vocab.id_to_piece(id))
    create_training_file(docs, vocab_list, args)


def build_pretrain_loader(path, vocab, n_batch):
    dataset = PregeneratedDataset(path, vocab, reduce_memory=True)
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=n_batch)
    return dataloader


if __name__ == '__main__':
    prefix = 8000
    demp_pretrain(f"../data/m_snli_{prefix}.model", f"../data/pretrain_bert_{prefix}_0")
    pass

