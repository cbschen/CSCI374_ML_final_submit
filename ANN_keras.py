#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:17:00 2019

@author: liujiachen
"""

import os
import sys
import random
import time 
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint


    



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

def recall(confusion_matrix,labels):
    recall_dict = {}
   
    for l in labels:
        idx = labels.index(l)
        numerator=confusion_matrix[idx][idx]
        if numerator == 0:
            recall = 0
        else:
            recall = confusion_matrix[idx][idx]/(sum(confusion_matrix[idx]))
        print("recall for "+str(l),recall)
        recall_dict[l]=recall
    best_recall = max(recall_dict, key = recall_dict.get)
    print("best recall",best_recall)
    worst_recall = min(recall_dict, key = recall_dict.get)
    print("worst recall",worst_recall)
    
def precision(confusion_matrix,labels):
    precision_dict = {}
    for l in labels:
        idx = labels.index(l)
        numerator=confusion_matrix[idx][idx]
        if numerator == 0:
            precision = 0
        else:
            precision = confusion_matrix[idx][idx]/(sum([confusion_matrix[row][idx]for row in range(len(confusion_matrix))]))
        print("precision for "+str(l),precision)
        precision_dict[l] = precision
    best_precision=max(precision_dict,key=precision_dict.get)
    print("best precision", best_precision)
    worst_precision=min(precision_dict,key=precision_dict.get)
    print("worst precision", worst_precision)
    
def print_matrix_CI(path,labels,confusion_matrix,test_accuracy,test_set):
    #print out confusion matrix    
    [print(l, end=", ") for l in labels] #print out all predicted labels
    print()
    #print out results with actual label
    for r in range(0, len(confusion_matrix)):
        [print(l, end=", ") for l in confusion_matrix[r]]
        print(labels[r])
        
    file_name = path.split("/")[-1]
    file_name = file_name.split(".csv")[0]
    
    
    outputfile = "./results_"+"ANN_keras_"+file_name+".csv"
    
    with open(outputfile, mode='w',newline='') as out:
        writer = csv.writer(out)
    
        writer.writerow([l for l in labels]+[""])
        for r in range(0, len(confusion_matrix)):
            writer.writerow([(str(l)) for l in confusion_matrix[r]]+[labels[r]])
    
    p = test_accuracy
    #print("accuracy",test_accuracy)
    n= test_set.shape[0] #number of test cases
    CI = [p - 2.24*((p*(1-p)/n))**0.5,p + 2.24*((p*(1-p)/n))**0.5]
    print("CI",CI)    



def run():
    
    start = time.time()
    
    path = "poem_new_attributes.csv" #The path to a file containing a data set (e.g., monks1.csv)
    seed = 30
    train_percent = 0.7
    val_percent = 0.15
    
    random.seed(seed)
    np.random.seed(seed)
    
    df = pd.read_csv(path) #read dataset in as a pandas dataframe
    
    print(df.shape)
    
    #check missing value
    print(df.isnull().values.any())
    
    
    #select subset(test purpose)
    #df = df [["Century","Word Diversity","Average Word Length","Number of Nouns","Number of Verbs"]]
    
    
    
    #shuffle dataset
    df = df.sample(frac=1, random_state = seed).reset_index(drop=True)
    
  
   
    
    #nominalization and one-hot encoding 
    columns_nominal = []
    for i in range(1,df.shape[1]):
        data = df.iloc[:,i]
        if data.dtype.kind in "iuf": #i singed int, u unsigned int, f float
            #it's numerical, normalize it
            diff = data.max()-data.min()
            if diff != 0:
                df.iloc[:,i] = (data-data.min())/(diff)
            else:
                df.iloc[:,i] = 0
        else:
            #nominal, record the columns for one hot encode
            columns_nominal.append(df.columns[i])
            df.iloc[:,i] = pd.Categorical(data)
            df_dummies = pd.get_dummies(df.iloc[:,i], prefix="category")
            

            df_onehot = df_dummies.drop(df_dummies.columns[-1], axis =1) #m-1 indicator variables
            df = pd.concat([df,df_onehot],axis=1)
            
           
    
    for col in columns_nominal: #drop the original columns
        df = df.drop(col, axis=1)
      
    shape = df.shape  # a tuple of the dataset's dimensions

        
    train_set,val_set, test_set = split_data(df,train_percent,val_percent,seed)
    
    
    
    
    train_set = train_set.to_numpy()
    test_set = test_set.to_numpy()
    val_set = val_set.to_numpy()
    dataset = df.to_numpy()
    
    labels = list(np.unique(dataset[:,0]))
    num_label = len(labels)
    
    X_train = train_set[:,1:]
    Y_train = train_set[:,0]
    
    
    
    dummy_Y_train = np.zeros((Y_train.shape[0],num_label))
    for i in range(0,Y_train.shape[0]):
        idx = labels.index(Y_train[i])
        dummy_Y_train[i,idx]=1
        
 
    
    
    

    X_test = test_set[:,1:]
    Y_test = test_set[:,0]
    
    dummy_Y_test = np.zeros((Y_test.shape[0],num_label))
    for i in range(0,Y_test.shape[0]):
        idx = labels.index(Y_test[i])
        dummy_Y_test[i,idx]=1
        
        
        
        
    X_val = val_set[:,1:]
    Y_val = val_set[:,0]
    
    dummy_Y_val = np.zeros((Y_val.shape[0],num_label))
    for i in range(0,Y_val.shape[0]):
        idx = labels.index(Y_val[i])
        dummy_Y_val[i,idx]=1
        
    
    
    
  
    model = tf.keras.Sequential()
    
    #a hidden layer with number of units to be half the input and output units
    model.add(layers.Dense(20, activation='relu',input_dim=shape[1]-1))
    
    
    # Add a softmax layer with output units:
    model.add(layers.Dense(num_label, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss='categorical_crossentropy',metrics=['accuracy'])
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=50),
             ModelCheckpoint(filepath='best_ANN_keras.h5', monitor='val_loss', save_best_only=True)]
    
    history=model.fit(X_train, dummy_Y_train, epochs=500, callbacks=callbacks,batch_size=128,validation_data = (X_val,dummy_Y_val))
    
    

    
    confusion_matrix = [[0]*num_label for m in range(0,num_label)]
    
    for instance in range(0, test_set.shape[0]): #iterate over test cases
        inputs = X_test[instance,:]
        
        
        
        target = dummy_Y_test[instance,:]
        
        
       
        
        
        prediction = model.predict_classes(inputs.reshape(1,shape[1]-1),batch_size=None,verbose=0)
        
        
        iactual = np.where(target==1)[0][0] #row index
        ipredict = prediction[0] #column index
        confusion_matrix[iactual][ipredict] += 1
    
    n_accurate_test=sum([confusion_matrix[idx][idx] for idx in range(len(confusion_matrix))])
    
    test_accuracy = n_accurate_test/test_set.shape[0]
    print("test accuracy", test_accuracy)
    
    print_matrix_CI(path,labels,confusion_matrix,test_accuracy,test_set)
    
    recall(confusion_matrix,labels)  
    precision(confusion_matrix,labels)
    end = time.time()
    print("run time", (end-start)/60,"min")       
run()