#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : TransE.py
# @Author: LauTrueYes
# @Date  : 2021-3-27 10:19
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, vocab_len, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.embed = nn.Embedding(num_embeddings=vocab_len, embedding_dim=config.embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=10, dim_feedforward=config.embed_dim )
        self.trans = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(config.embed_dim, config.num_classes) #分类
        self.ln = nn.LayerNorm(config.num_classes)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, word_ids, label_ids=None):
        """

        :param word_ids: batch_size * max_seq_len
        :param label_ids: batch_size
        :return:
        """
        emb = self.embed(word_ids.permute(1,0))
        x = self.trans(emb)
        x = x[0]
        x = self.fc(x)
        x = self.ln(x)
        label_predict = x
        if label_ids != None:
            loss = self.loss_fct(label_predict, label_ids)
        else:
            loss = None

        return loss, label_predict.argmax(dim=-1)