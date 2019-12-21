#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 00:25:40 2019

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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
import bidirectional_LSTM_keras_gpu

def main():
    
    file_name = "poems_reordered_further_cleaned.csv"
    results = {} #key is accuracy, value is a list of hyperparameters
    for embedding in [64, 128,256,512]:
        for n_neuron in [128,256,512]:
            for lrate in [0.0001, 0.001, 0.01]:
                for remove in ["T","F"]:
                    for n_layer in [1,2,3]:
                        accuracy = bidirectional_LSTM_keras_gpu.main(embedding,n_neuron,lrate,remove,n_layer)
                        results[accuracy] = [embedding,n_neuron,lrate,remove,n_layer]
                
    outputfile = "./grid_results_"+"bidirect_LSTM_"+file_name+".csv"
    
    with open(outputfile, mode='w',newline='') as out:
        writer = csv.writer(out)
    
        writer.writerow(["accuracy","embedding","n_neuron","lrate","remove_stop_words","n_rec_layer"])
        for accuracy in results:
            writer.writerow([accuracy]+results[accuracy])            
    best_accuracy = max(results, key = results.get)
    print("best accuracy",best_accuracy,"settings",results[best_accuracy])
    pprint(results)            

main()