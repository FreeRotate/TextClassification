#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: LauTrueYes
# @Date  : 2021-11-28
import torch
import pandas as pd

class ContentLabel(object):
    def __init__(self, content, label):
        self.content = content
        self.label = label
    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


def load_dataset(file_path, test_file=False):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            line = line.strip('\n')
            content, label = line.split('\t')
            dataset.append(ContentLabel(content, label))
    return dataset


class Vocab(object):
    def __init__(self):
        self.id2word = None
        self.word2id = {'PAD':0}

    def add(self, dataset, test_file=False):
        id = len(self.word2id)
        for item in dataset:
            for word in item.content:
                if word not in self.word2id:
                    self.word2id.update({word: id})
                    id += 1
        self.id2word = {j: i for i, j in self.word2id.items()}
    def __len__(self):
        return len(self.word2id)


class DataLoader(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for index in range(len(self.dataset)):
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch):
            yield batch
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def batch_variable(batch_data, vocab, config):
    batch_size = len(batch_data)
    max_seq_len = config.max_seq
    word_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_ids = torch.zeros((batch_size), dtype=torch.long)

    for index, cl in enumerate(batch_data):
        seq_len = len(cl.content)
        if seq_len > max_seq_len:
            cl.content = cl.content[:max_seq_len]
            word_ids[index, :max_seq_len] = torch.tensor([vocab.word2id[item] for item in cl.content])
        else:
            word_ids[index, :seq_len] = torch.tensor([vocab.word2id[item] for item in cl.content])
        label_ids[index] = torch.tensor([int(cl.label)])

    return word_ids.to(config.device), label_ids.to(config.device)
