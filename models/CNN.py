#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : CNN.py
# @Author: LauTrueYes
# @Date  : 2021-3-11 21:22
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, vocab_len, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.embed = nn.Embedding(num_embeddings=vocab_len, embedding_dim=config.embed_dim)
        self.convs = nn.ModuleList(
            [nn.Sequential
                (
                nn.Conv1d(in_channels=config.embed_dim, out_channels=config.kernel_nums[i]
                          , padding=ks // 2, kernel_size=ks),
                nn.LeakyReLU(),
                nn.AdaptiveMaxPool1d(output_size=1)
            )
                for i, ks in enumerate(config.kernal_sizes)]
        )  # 卷积
        self.out_size = sum([i for i in config.kernel_nums])
        self.fc = nn.Linear(self.out_size, self.num_classes)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, word_ids, label_ids=None):
        """

        :param word_ids: batch_size * max_seq_len
        :param label_ids: batch_size
        :return:
        """
        x = self.embed(word_ids)
        x = x.permute(0,2,1)
        x = [conv(x).squeeze(-1) for conv in self.convs]
        x = torch.cat(tuple(x), dim=-1).contiguous()
        x = self.fc(x)
        label_predict = x
        if label_ids != None:
            loss = self.loss_fct(label_predict, label_ids)
        else:
            loss = None

        return loss, label_predict.argmax(dim=-1)