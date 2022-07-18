#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : predict.py
# @Author: LauTrueYes
# @Date  : 2021-11-28
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import batch_variable, ContentLabel


def predict(model, test_loader, config, vocab):
    model.load_state_dict(torch.load(config.save_path), False)
    content, labels = [], []
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(test_loader)):
            word_ids, _ = batch_variable(batch_data, vocab, config)
            _, logits = model(word_ids)

            for item, label in zip(batch_data, logits.data):
                content.append(item.content)
                labels.append(config.id2class[label.data.item()])
    dict = {'标题':content, '类别':labels}
    file = pd.DataFrame(dict, columns=[key for key in dict.keys()])
    file.to_csv(config.predict_path, index=False, encoding='utf_8_sig')
