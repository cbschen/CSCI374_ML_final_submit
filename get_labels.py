#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:05:30 2019

@author: liujiachen
"""

import os
import sys
import random
import time 
import numpy as np
import pandas as pd

#this is a script to get number of attributes, the number of possible labels, and the proportion of each label in the data set

def main():
    paths = ["poems_reordered_further_cleaned.csv"]
    for path in paths:
        print("==============================================================")
        print(path)
        df = pd.read_csv(path)
        print("dataset shape:", df.shape)
        print("number of attributes",df.shape[1]-1)
        labels = np.unique(df.iloc[:,0].to_numpy())
        print("labels:",labels)
        print("number of labels",len(labels))
        label_freq={}
        for i in range (0,df.shape[0]):
            label = df.iloc[i,0]
            if label not in label_freq:
                label_freq[label] = 1
            else:
                label_freq[label] = label_freq[label] + 1
        for label in labels:
            print("proportion for",label,"is",label_freq[label]/df.shape[0])
        print("==============================================================")

main()