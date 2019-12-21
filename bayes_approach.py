#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:23:18 2019

no stemming
no stopwords removal


@author: liujiachen
"""

import StemmingUtil
import math
import os
import sys
import random
import time 
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from pprint import pprint 
import copy
import nltk
from nltk.corpus import stopwords
from string import punctuation

def split_data(data, tr_size, val_size, seed):
    """Divides into a training set, validation set, and test set.
    tr_size is % of total to make the training set, and val_size is % of the remainder to make the validation set.
    Returns the training, validation, and test sets as pandas dataframes. """
    tr_set = data.sample(frac=tr_size, random_state=seed) #get a train set from data
    remain_data = data.drop(tr_set.index).reset_index(drop=True) #drop train set rows from data and reset remain_data index
    tr_set = tr_set.reset_index(drop=True) #reset train set index
    val_set = remain_data.sample(frac=(val_size*data.shape[0])/remain_data.shape[0], random_state=seed) #get a val set from remain set
    test_set = remain_data.drop(val_set.index).reset_index(drop=True) #remove val set rows from remain set to obtain test set and reset test set index 
    val_set = val_set.reset_index(drop=True)
    return tr_set, val_set, test_set


def naive_bayes(passage,author, WAP, p_year): 
    #passage is a list of unique words in a passage
    #each author is a label
    #calculate P(given label)*p(combination of observed attributes|this label)
   
    val = math.log(p_year[author])
    for w in passage:
        val = val + math.log(WAP[w][author])
    return val


def main():
    start = time.time()
    seed = 30
    random.seed(seed)
    np.random.seed(seed)
    train_percent = 0.7
    val_percent = 0.15 #we will not use validation, but just to partition it out so that we can compare with neural net fairly
    
    
    stemming = True
    remove_stop_words = True
    
    
    df = pd.read_csv("poems_reordered_further_cleaned.csv")
    
    if not remove_stop_words:
        words_to_remove =  ["\n","\s"]
    else:
        words_to_remove =  stopwords.words("english")+["\n","\s"]
    
    
    for i in range (0, df.shape[0]):
        line = df.iloc[i,1]
        
        words = StemmingUtil.parseTokens(line)
        words=[w for w in words if w not in words_to_remove]
        if stemming:
            words = StemmingUtil.createStems(words)
        newline = " ".join(words)
        df.iloc[i,1] = newline
    
    
    
    
    
    
    #shuffle dataset
    df = df.sample(frac=1, random_state = seed).reset_index(drop=True)
    train_set,val_set,test_set = split_data(df,train_percent,val_percent,seed)
    
    print("train_set shape:",train_set.shape)
    print("test_set shape:",test_set.shape)
    print("val_set shape:",val_set.shape)
    
    
    
    books = {}
    
    for i in range (train_set.shape[0]):
        year = train_set.iloc[i,0]
        if year not in books:
            books[year] = train_set.iloc[i,1]
        else:
            books[year] = books[year]+train_set.iloc[i,1]
            
    all_words=[]    
    punc = list(punctuation)+["“","”"] #seems the punctuation in text file has a different font 
    
   
    
    
    WAP = {} #a dict of dict, {word:{author:p(word|author)}} for the training set
    
    
             
    #author means year in this case         
    for author in books:        
        
        line = books[author]
         
        
        words = StemmingUtil.parseTokens(line)
        words=[w for w in words if w not in words_to_remove]
       
        books[author] = words
        all_words = all_words + words
       
    #now we changed the value for books from a string of passage to a list of preprocessed words
    
    
    #####now we load test set############   
    test_cases = []
   
    year_freq = {} #frequency of each century
    for i in range (test_set.shape[0]):
        year = test_set.iloc[i,0] #year means century
        if year in year_freq:
            year_freq[year] +=1
        else:
            year_freq[year] = 1
        passage = test_set.iloc[i,1]
        tokenized = StemmingUtil.parseTokens(passage)
        tokenized=[w for w in tokenized if w not in words_to_remove]
        test_cases.append([year,tokenized])
        all_words = all_words + tokenized
        
    p_year={} #probability of each century
    for year in year_freq:
        p_year[year] = year_freq[year]/test_set.shape[0]
    
    
                    
    
    all_words = set(all_words) #convert to set, we have a set of all possible words
    b=len(all_words)
    
    for w in all_words:
        WAP[w]={}
    
    for author in books:
        
        freq={} #key is unique word, value is freq of word 
        for word in books[author]:
            if word not in freq.keys():
                freq[word]=1
            else:
                freq[word] = freq[word] +1
        n_words = len(books[author])
        bag = set(books[author])
       
        
        total = n_words+b
        
        
        for word in all_words:
            
            if word not in bag:
               
                
                p = 1/total
                WAP[word][author]=p #value is P(word|author) with pesudo counts
                
            else:
               
                p = (freq[word]+1)/total
                WAP[word][author]=p
               
 
    
    
    
    labels = [item[0] for item in test_cases] 
    labels = list(set(labels))
    
    num_label = len(labels)
    confusion_matrix = [[0]*num_label for m in range(0,num_label)]
               
    
    for case in test_cases:
        author = case[0]
        passage = case[1]
        results = {}
        for potential_author in labels:
            results[potential_author]=naive_bayes(passage,potential_author,WAP,p_year)
        prediction = max(results,key=results.get)
        
        
        iactual = labels.index(author) #row index
        ipredict = labels.index(prediction) #column index
        confusion_matrix[iactual][ipredict] += 1
        
     

        
    n_accurate_test=sum([confusion_matrix[idx][idx] for idx in range(len(confusion_matrix))]) 
    test_accuracy = n_accurate_test/len(test_cases)
    print("test accuracy", test_accuracy)
    
    recall_dict = {}
   
    for author in labels:
        idx = labels.index(author)
        numerator=confusion_matrix[idx][idx]
        if numerator == 0:
            recall = 0
        else:
            recall = confusion_matrix[idx][idx]/(sum(confusion_matrix[idx]))
        print("recall for "+str(author),recall)
        recall_dict[author]=recall
    best_recall = max(recall_dict, key = recall_dict.get)
    print("author with best recall",best_recall)
    worst_recall = min(recall_dict, key = recall_dict.get)
    print("author with worst recall",worst_recall)
    
    
    
    precision_dict = {}
    for author in labels:
        idx = labels.index(author)
        numerator=confusion_matrix[idx][idx]
        if numerator == 0:
            precision = 0
        else:
            precision = confusion_matrix[idx][idx]/(sum([confusion_matrix[row][idx]for row in range(len(confusion_matrix))]))
        print("precision for "+str(author),precision)
        precision_dict[author] = precision
    best_precision=max(precision_dict,key=precision_dict.get)
    print("author with best precision", best_precision)
    worst_precision=min(precision_dict,key=precision_dict.get)
    print("author with worst precision", worst_precision)    
    
    

    
    
    #print out confusion matrix    
    [print(l, end=", ") for l in labels] #print out all predicted labels
    print()
    #print out results with actual label
    for r in range(0, len(confusion_matrix)):
        [print(l, end=", ") for l in confusion_matrix[r]]
        print(labels[r])
    
    
    
    outputfile = "./results_authors.csv"
    
    with open(outputfile, mode='w',newline='') as out:
        writer = csv.writer(out)
    
        writer.writerow([l for l in labels]+[""])
        for r in range(0, len(confusion_matrix)):
            writer.writerow([(str(l)) for l in confusion_matrix[r]]+[labels[r]])
    
    p = test_accuracy
    print("accuracy",test_accuracy)
    n= len(test_cases) #number of test cases
    CI = [p - 1.96*((p*(1-p)/n))**0.5,p + 1.96*((p*(1-p)/n))**0.5]
    print("CI",CI)           
    
    
    
    
    end = time.time()
    print("run time", (end-start)/60,"min")

main()
        
    
    
    
    
            





       
            
    
    
    
    
    
    

