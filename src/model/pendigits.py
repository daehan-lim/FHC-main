#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/27 14:38
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : mnist_LeNet.py
# @Software  : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

class pendigits(nn.Module):

    def __init__(self):
        super(pendigits,self).__init__()

        # 16 32 64 128 9
        self.inn = nn.Linear(28*28,32, bias=True)
        self.fc2 = nn.Linear(32,64, bias=True)
        self.fc3 = nn.Linear(64,128, bias=True)
        self.out = nn.Linear(128, 10, bias=True)


        self.testin = nn.Linear(28*28,512, bias=True)
        self.testfc2 = nn.Linear(512,128, bias=True)
        self.testout = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.testin(x)
        x = F.relu(x)
        x = self.testfc2(x)
        x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        #x = self.out(x)

        x=self.testout(x)
        x = F.relu(x)
        self.softmax = F.softmax(x)
        return x