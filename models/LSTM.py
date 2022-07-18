#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : LSTM.py
# @Author: LauTrueYes
# @Date  : 2021-11-28 10:19
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_len, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.embed = nn.Embedding(num_embeddings=vocab_len, embedding_dim=config.embed_dim)
        self.lstm = nn.LSTM(input_size=config.embed_dim, hidden_size=config.embed_dim, bidirectional=True)
        self.fc = nn.Linear(config.embed_dim * 2, config.num_classes) #分类

        self.ln = nn.LayerNorm(config.num_classes)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, word_ids, label_ids=None):
        """

        :param word_ids: batch_size * max_seq_len
        :param label_ids: batch_size
        :return:
        """
        x = self.embed(word_ids.permute(1,0))
        x, _ = self.lstm(x)
        x = x[-1]
        x = self.fc(x)
        x = self.ln(x)
        label_predict = x
        if label_ids != None:
            loss = self.loss_fct(label_predict, label_ids)
        else:
            loss = None

        return loss, label_predict.argmax(dim=-1)
