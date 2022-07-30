#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: LauTrueYes
# @Date  : 2021-11-25
import torch
import numpy as np
import torch.optim as optim
from utils import batch_variable
from sklearn import metrics

def train(model, train_loader, dev_loader, config, vocab):

    loss_all = np.array([], dtype=float)
    label_all = np.array([], dtype=float)
    predict_all = np.array([], dtype=float)
    dev_best_f1 = float('-inf')

    optimizer = optim.AdamW(params=model.parameters(), lr=config.lr)
    for epoch in range(0, config.epochs):
        for batch_idx, batch_data in enumerate(train_loader):
            model.train()   #训练模型
            word_ids, label_ids = batch_variable(batch_data, vocab, config)
            loss, label_predict = model(word_ids, label_ids)

            loss_all = np.append(loss_all, loss.data.item())
            label_all = np.append(label_all, label_ids.data.cpu().numpy())
            predict_all = np.append(predict_all, label_predict.data.cpu().numpy())
            acc = metrics.accuracy_score(predict_all, label_all)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print("Epoch:{}--------Iter:{}--------train_loss:{:.3f}--------train_acc:{:.3f}".format(epoch+1, batch_idx+1, loss_all.mean(), acc))
        dev_loss, dev_acc, dev_f1, dev_report, dev_confusion = evaluate(model, dev_loader, config, vocab)
        msg = "Dev Loss:{}--------Dev Acc:{}--------Dev F1:{}"
        print(msg.format(dev_loss, dev_acc, dev_f1))
        print("Dev Report")
        print(dev_report)
        print("Dev Confusion")
        print(dev_confusion)

        if dev_best_f1 < dev_f1:
            dev_best_f1 = dev_f1
            torch.save(model.state_dict(), config.save_path)
            print("***************************** Save Model *****************************")


def evaluate(config, model, dev_loader, vocab):
    model.eval()    #评价模式
    loss_all = np.array([], dtype=float)
    predict_all = np.array([], dtype=int)
    label_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dev_loader):
            word_ids, label_ids = batch_variable(batch_data, vocab, config)
            loss, label_predict = model(word_ids, label_ids)

            loss_all = np.append(loss_all, loss.data.item())
            predict_all = np.append(predict_all, label_predict.data.cpu().numpy())
            label_all = np.append(label_all, label_ids.data.cpu().numpy())
    acc = metrics.accuracy_score(label_all, predict_all)
    f1 = metrics.f1_score(label_all, predict_all, average='macro')
    report = metrics.classification_report(label_all, predict_all, target_names=config.class_list, digits=3)
    confusion = metrics.confusion_matrix(label_all, predict_all)

    return loss.mean(), acc, f1, report, confusion













