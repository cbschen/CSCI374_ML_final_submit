#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:16:05 2019

@author: liujiachen
"""
import StemmingUtil
import os
import sys
import random
import time 
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from collections import Counter 

def main():
    
    remove_stop_words = "F"
    stemming = True
    df = pd.read_csv("poems_reordered_further_cleaned.csv")
    
    #col =  df["Number of Words"]
    
    #x = [1,2,3,4,5]
    #plt.hist(col,bins = 1000,range = (0,1000))
    #plt.show()
    
    nltk.download('stopwords')
    nltk.download('punkt')
    start = time.time()
    seed = 30
    random.seed(seed)
    np.random.seed(seed)
    train_percent = 0.7
    val_percent = 0.15
    
    if remove_stop_words == "F":
        words_to_remove =  ["\n","\s"]
    else:
        words_to_remove =  stopwords.words("english")+["\n","\s"]
    
    path = "poems_reordered_further_cleaned.csv"
    
    df = pd.read_csv(path)
    
    
    
    
    #shuffle dataset
    df = df.sample(frac=1, random_state = seed).reset_index(drop=True)
    
    
    
    all_words =[]
    
    
    count = 0

    max = 0

    for i in range (0, df.shape[0]):
        line = df.iloc[i,1]
        
        
        
        words = StemmingUtil.parseTokens(line)
        words=[w for w in words if w not in words_to_remove]
        
        if stemming:
            words = StemmingUtil.createStems(words)
        
        newline = " ".join(words)
        df.iloc[i,1] = newline
        count += len(words)
        
        if len(words)>max:
          max = len(words)

        #print(len(words))
        
        
            
        
        all_words = all_words + words
        
    average_len = int(count/df.shape[0])    
    #all_words = set(all_words)
    vocab_size = len(set(all_words))
    
    
    ###########
    N = vocab_size
    
    topN = Counter(all_words).most_common(vocab_size)
    print(topN)
    
    topN = [item[0] for item in topN]
    
    outputfile = "./poem_new_attributes.csv"
    with open(outputfile, mode='w',newline='') as out:
        writer = csv.writer(out)
    
        writer.writerow(["century"]+[l for l in topN])
        
        for i in range (0, df.shape[0]):
            line = df.iloc[i,1]
            
            words = StemmingUtil.parseTokens(line)
            
            length = len(words)
            
            freq_count = {} #key is each unique work, value is freq
            attributes = []
            
            
            for word in words:
                if word not in freq_count:
                    freq_count[word] = 1
                else:
                    freq_count[word] +=1
            for word in topN:
                if word in freq_count:
                    prop = freq_count[word]/length
                else:
                    prop = 0
                attributes.append(prop)
            writer.writerow([df.iloc[i,0]]+attributes)
        
    

main()