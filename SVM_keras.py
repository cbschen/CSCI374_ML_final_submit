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
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 

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
    
def print_matrix_CI(path,labels,confusion_matrix,test_accuracy,setting,X_test):
    #print out confusion matrix    
    [print(l, end=", ") for l in labels] #print out all predicted labels
    print()
    #print out results with actual label
    for r in range(0, len(confusion_matrix)):
        [print(l, end=", ") for l in confusion_matrix[r]]
        print(labels[r])
        
    file_name = path.split("/")[-1]
    file_name = file_name.split(".csv")[0]
    
    
    outputfile = "./results_"+"SVM_"+file_name+setting+".csv"
    
    with open(outputfile, mode='w',newline='') as out:
        writer = csv.writer(out)
    
        writer.writerow([l for l in labels]+[""])
        for r in range(0, len(confusion_matrix)):
            writer.writerow([(str(l)) for l in confusion_matrix[r]]+[labels[r]])
    
    p = test_accuracy
    #print("accuracy",test_accuracy)
    n= X_test.shape[0] #number of test cases
    CI = [p - 1.96*((p*(1-p)/n))**0.5,p + 1.96*((p*(1-p)/n))**0.5]
    print("CI",CI) 

def plot_cloud(label,df): #plot out word cloud
  words = ""
  for poem in df[df["century"]==label]["poem"]:
    words = words + poem + " "
  cloud = WordCloud(width = 600, height = 600).generate(words)
  plt.imshow(cloud)
  plt.axis("off")
  plt.title(str(label)+"th century")
  plt.show()

def main():

  path = "poems_reordered_further_cleaned.csv"
  nltk.download('stopwords')
  nltk.download('punkt')
  start = time.time()
  seed = 30
  random.seed(seed)
  np.random.seed(seed)
  tf.compat.v1.set_random_seed(seed)
  remove_stop_words = "T"
  if remove_stop_words == "F":
        words_to_remove =  ["\n","\s"]
  else:
        words_to_remove =  stopwords.words("english")+["\n","\s"]
  stemming = True
  df = pd.read_csv(path)
  #shuffle dataset
  df = df.sample(frac=1, random_state = seed).reset_index(drop=True)
  df["century"]=df["century"].map({17:"17",18:"18",19:"19"})
  Y = df["century"].values

  for i in range (0, df.shape[0]):
          line = df.iloc[i,1]
          
          words = StemmingUtil.parseTokens(line)
          words=[w for w in words if w not in words_to_remove]
          
          if stemming:
              words = StemmingUtil.createStems(words)
          
          newline = " ".join(words)
          df.iloc[i,1] =  newline

  
  vectorizer = TfidfVectorizer()
  X= vectorizer.fit_transform(df["poem"])

  X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

  model = SVC(kernel = "linear", C=2.0)
  model.fit(X_train, Y_train)


  dataset = df.to_numpy()
    
  labels = list(np.unique(dataset[:,0]))
  num_label = len(labels)
  #run test set
  confusion_matrix = [[0]*num_label for m in range(0,num_label)]
  
  for instance in range(0, Y_test.shape[0]): #iterate over test cases
      inputs = X_test[instance,:]
      
      
      
      target = Y_test[instance]
      
      
      #rint(target)
      
      
      prediction = model.predict(inputs)
      
      #print(prediction[0])
     
      
      iactual = labels.index(target) #row index
      ipredict = labels.index(prediction) #column index
      confusion_matrix[iactual][ipredict] += 1
  
  n_accurate_test=sum([confusion_matrix[idx][idx] for idx in range(len(confusion_matrix))])
  
  test_accuracy = n_accurate_test/Y_test.shape[0]
  
  print("setting:","remove_stop_words",str(remove_stop_words))
  
  print("test accuracy", test_accuracy)
  
  setting = "remove_"+str(remove_stop_words)
  
  print_matrix_CI(path,labels,confusion_matrix,test_accuracy,setting,X_test)
  
  recall(confusion_matrix,labels)  
  
  precision(confusion_matrix,labels)
  
  plot_cloud("17",df)
  plot_cloud("18",df)
  plot_cloud("19",df)
  
  
  
  
  end = time.time()
  
  print("run time", (end-start)/60,"min")
  
  #setting = "embed_"+str(embedding_dim)+"neuron_"+str(n_neuron)+"lrate_"+str(lrate)+"remove_"+str(remove_stop_words)+"layer_"+str(n_rec_layer)
  
  
  
  return test_accuracy
main()


