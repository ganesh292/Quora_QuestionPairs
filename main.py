# -*- coding: utf-8 -*-
"""Architecture+Embeddings_Ganesh.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10a_xNH0kyI0xMAAor1mjn8_Dp9E7WFJ0
"""
# avoid decoding problems
import sys
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
from bert_serving.client import BertClient

from numpy import genfromtxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec, FastText
from fse.models import Sentence2Vec
# Make sure, that the fast version of fse is available!
from fse.models.sentence2vec import CY_ROUTINES
assert CY_ROUTINES

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import callbacks
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers, Add, concatenate, Layer,Lambda
from keras.optimizers import RMSprop, SGD, Adam

# Network Architecture
from keras.models import load_model

from keras.models import Sequential, Model
from keras.layers import Conv1D , MaxPooling1D, Flatten,Dense,Input,Lambda
from keras.layers import LSTM, Concatenate, Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras import backend as K
import tensorflow as tf

import numpy as np

import torch
from models import InferSent
from keras import regularizers




def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

#Word Embeddings
def infersent_glove():
    #Set Model for InferSent+Glove
    V = 1
    MODEL_PATH = '/tmp/GloVe/encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    modelg = InferSent(params_model)
    modelg.load_state_dict(torch.load(MODEL_PATH))
    # Keep it on CPU or put it on GPU
    use_cuda = True
    modelg = modelg.cuda() if use_cuda else modelg

    # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
    W2V_PATH = '/tmp/GloVe/glove.840B.300d.txt' if V == 1 else '/home/ganesh/Quora_dev/tmp/GloVe/glove.840B.300d.txt'
    modelg.set_w2v_path(W2V_PATH)
    # Load embeddings of K most frequent words
    modelg.build_vocab_k_words(K=100000)
    return modelg

modelg = infersent_glove()
def get_fastext(sentences_tok):
  print("Training FastText model ...\n")
  model = FastText(size=324, window=10, min_count=1)  # instantiate
  model.build_vocab(sentences_tok)
  model.train(sentences=sentences_tok,total_examples=len(sentences_tok),epochs=5)  # train
  se = Sentence2Vec(model)
  ft_embeddings = se.train(sentences_tok)
  return ft_embeddings
  

def get_w2v(sentences_tok):
  #Word2Vec Embeddings
  print("Training W2v model ...\n")
  w2v_model = Word2Vec(sentences_tok,size=324,window=10, min_count=1)
  se = Sentence2Vec(w2v_model)
  w2v_embeddings = se.train(sentences_tok)
  return w2v_embeddings

def get_glove(sentences):
  print("Training glove+infersent model ...\n")
  embeddings = modelg.encode(sentences, bsize=128, tokenize=False, verbose=True)
  pca = PCA(n_components=324) #reduce down to 50 dim
  glove_embeddings = pca.fit_transform(embeddings)
  return glove_embeddings

def get_bertembeddings(q1,q2):
  q = list(map(lambda x, y: x+ ' ||| ' +y, q1, q2))
  bc = BertClient()
  bert_embeddings = bc.encode(q)
  return bert_embeddings


def create_base_network_cnn(input_dimensions):
  
  input  = Input(shape=(input_dimensions[0],input_dimensions[1]))
  conv1  = Conv1D(filters=32,kernel_size=8,strides=1,activation = 'relu',name='conv1')(input)
  pool1  = MaxPooling1D(pool_size=1,strides=1,name='pool1')(conv1)
  conv2  = Conv1D(filters=64,kernel_size=6,strides=1,activation='relu',name='conv2')(pool1)
  pool2  = MaxPooling1D(pool_size=1,strides=1,name='pool2')(conv2)
  conv3  = Conv1D(filters=128,kernel_size=4,strides=1,activation='relu',name='conv3')(pool2)
  flat   = Flatten(name='flat_cnn')(conv3)
  # dense  = Dense(376,name='dense_cnn')(flat)
  dense  = Dense(100,name='dense_cnn')(flat)
   
  model  = Model(input=input,output=dense)
  return model

def create_base_network_lstm(input_dimensions):
  input = Input(shape=(input_dimensions[0],1))
  layer1 = LSTM(20, return_sequences=True,activation='relu',name='lstm_1')(input)
  layer2 = LSTM(20,return_sequences=False,activation='relu',name='lstm_2')(layer1)
  dense = Dense(100,name='dense_lstm')(layer2)
  
  model = Model(input=input,output=dense)
  return model
  

def dense_network(features):
  input = Input(shape=(1,features[0],features[1]))
  #x = Flatten()(features)
  d1 = Dense(128, activation='relu')(input)
  drop1 = Dropout(0.1)(d1)
  d2 = Dense(128, activation='relu')(drop1)
  drop2 = Dropout(0.1)(d2)
  d3 = Dense(2, activation='relu')(drop2)
  model = Model(input = input,output=d3)
  return model
  

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

    
def add_features():
  if sys.argv[1] == "b":
    data_features = pd.read_csv("quora_features_BERT_balanced.csv")
  else:
    data_features = pd.read_csv("quora_features_BERT_unbalanced.csv")
  
  features = data_features.drop(['question1', 'question2', 'is_duplicate','jaccard_distance'],axis=1).values
  print('Shape of Features added',features.shape)
  return features

def create_network(input_dimensions,num_features):

  # #Fasttext
  base_network_lstm_1 = create_base_network_lstm(input_dimensions)
  input_a_lstm_1 = Input(shape=(input_dimensions[0],1))
  input_b_lstm_1 = Input(shape=(input_dimensions[0],1))
  # LSTM with embedding 1
  inter_a_lstm_1 = base_network_lstm_1(input_a_lstm_1)
  inter_b_lstm_1 = base_network_lstm_1(input_b_lstm_1)
  d_lstm_1 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_1, inter_b_lstm_1])


  #W2V
  base_network_lstm_2 = create_base_network_lstm(input_dimensions)
  input_a_lstm_2 = Input(shape=(input_dimensions[0],1))
  input_b_lstm_2 = Input(shape=(input_dimensions[0],1))
  # LSTM with embedding 2
  inter_a_lstm_2 = base_network_lstm_2(input_a_lstm_2)
  inter_b_lstm_2 = base_network_lstm_2(input_b_lstm_2)
  d_lstm_2 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_2, inter_b_lstm_2])


  #Glove
  base_network_lstm_3 = create_base_network_lstm(input_dimensions)
  input_a_lstm_3 = Input(shape=(input_dimensions[0],1))
  input_b_lstm_3 = Input(shape=(input_dimensions[0],1))
  # LSTM with embedding 3
  inter_a_lstm_3 = base_network_lstm_3(input_a_lstm_3)
  inter_b_lstm_3 = base_network_lstm_3(input_b_lstm_3)
  d_lstm_3 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_3, inter_b_lstm_3])

  #BERT
  base_network_lstm_4 = dense_network([768,1])
  input_a_lstm_4 = Input(shape=(1,768,1))
  input_b_lstm_4 = Input(shape=(1,768,1))
   # LSTM with embedding 3
  inter_a_lstm_4 = base_network_lstm_4(input_a_lstm_4)
  inter_b_lstm_4 = base_network_lstm_4(input_b_lstm_4)

  
  #CNN
  base_network_cnn = create_base_network_cnn(input_dimensions)
  # CNN with 3 channel embedding
  input_a_cnn = Input(shape=(input_dimensions[0],input_dimensions[1]))
  input_b_cnn = Input(shape=(input_dimensions[0],input_dimensions[1]))
  inter_a_cnn = base_network_cnn(input_a_cnn)
  inter_b_cnn = base_network_cnn(input_b_cnn)



  # d_cnn = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_cnn, inter_b_cnn])
  d_lstm_1 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_1, inter_b_lstm_1])
  d_lstm_2 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_2, inter_b_lstm_2])
  d_lstm_3 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_3, inter_b_lstm_3])
  d_lstm_4 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_4, inter_b_lstm_4])
  
  # Additional Features from Thakur (BERT)
  features = Input(shape=(num_features,))

  #BERT itself
  features_b = Input(shape=(768,))
  
  
  #Concatenation of Features
  feature_set = Concatenate(axis=-1)([d_lstm_1,d_lstm_2,d_lstm_3,features,features_b])
  # feature_set = Concatenate(axis=-1)([d_cnn,d_lstm_1,d_lstm_2,d_lstm_3,features,features_b])
  # feature_set = Concatenate(axis=-1)([d_cnn,d_lstm_4,features,features_b])
  # feature_set = add_features(feature_set)

  #Final Dense Layer
  d1 = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.1))(feature_set)
  drop1 = Dropout(0.3)(d1)
  d2 = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.1))(drop1)
  drop2 = Dropout(0.3)(d2)
  d3 = Dense(1, activation='sigmoid')(drop2)

  # model = Model(input=[input_a_cnn, input_b_cnn , input_a_lstm_4, input_b_lstm_4,features,features_b], output=d3)
  # model = Model(input=[input_a_cnn, input_b_cnn , input_a_lstm_1, input_b_lstm_1, input_a_lstm_2, input_b_lstm_2, input_a_lstm_3, input_b_lstm_3,features,features_b], output=d3)
  model = Model(input=[input_a_lstm_1, input_b_lstm_1, input_a_lstm_2, input_b_lstm_2, input_a_lstm_3, input_b_lstm_3,input_a_lstm_4, input_b_lstm_4,features,features_b], output=d3)

  print("Model Architecture Designed")
  return model
  



def main():
  #default params
  print('Parameters I/p: u/b embtrain/noembtrain train/test resume/new','modelname')
  if sys.argv[1] == 'o':
    del sys.argv[1:2]
  else:
    sys.argv[1:] = ['b','noembtrain','train','new','QQP_08_0.6723.h5']

    #Get Dataset
  if sys.argv[1] == "u":
    df_sub = pd.read_csv('data_unbalanced.csv')
    print('Shape of unbalanced Dataset',df_sub.shape)
  elif sys.argv[1] == "b":
    df_sub = pd.read_csv('data_balanced.csv')
    print('Shape of Balanced Dataset',df_sub.shape)
  else:
    return 0
  
  df_sub['question1'] = df_sub['question1'].apply(lambda x: str(x))
  df_sub['question2'] = df_sub['question2'].apply(lambda x: str(x))
  q1sents = list(df_sub['question1'])
  q2sents = list(df_sub['question2'])
  tokenized_q1sents = [word_tokenize(i) for i in list(df_sub['question1'])]
  tokenized_q2sents = [word_tokenize(i) for i in list(df_sub['question2'])]

  #Compute Embeddings for Q1 Pair
  if sys.argv[2] == 'embtrain':
    ft_emb_q1 = get_fastext(tokenized_q1sents)
    w2v_emb_q1 = get_w2v(tokenized_q1sents)
    glove_emb_q1 = get_glove(q1sents)


    #Compute Embeddings for Q2 Pair
    ft_emb_q2 = get_fastext(tokenized_q2sents)
    w2v_emb_q2 = get_w2v(tokenized_q2sents)
    glove_emb_q2 = get_glove(q2sents)
  else:
    if sys.argv[1] == "u":
      print('Loading Embeddings W2vec')
      w2v_emb_q1 = genfromtxt('/tmp/Ganesh_MSCI/Unbalanced_Embeddings/word2vec/w2vec_q1_unbalanced.csv', delimiter=',',skip_header=1)
      w2v_emb_q2 = genfromtxt('/tmp/Ganesh_MSCI/Unbalanced_Embeddings/word2vec/w2vec_q2_unbalanced.csv', delimiter=',',skip_header=1)
      w2v_emb_q1 = np.delete(w2v_emb_q1, 0, 1)
      w2v_emb_q2 = np.delete(w2v_emb_q2, 0, 1)
      print('Loading Embeddings fastext')
      ft_emb_q1 = genfromtxt('/tmp/Ganesh_MSCI/Unbalanced_Embeddings/fastext/fastext_q1_unbalanced.csv', delimiter=',',skip_header=1)
      ft_emb_q2 = genfromtxt('/tmp/Ganesh_MSCI/Unbalanced_Embeddings/fastext/fastext_q2_unbalanced.csv', delimiter=',',skip_header=1)
      ft_emb_q1 = np.delete(ft_emb_q1, 0, 1)
      ft_emb_q2 = np.delete(ft_emb_q2,0, 1)
      print('Loading Embeddings glove')
      glove_emb_q1 = genfromtxt('/tmp/Ganesh_MSCI/Unbalanced_Embeddings/glove/glove_q1_unbalanced.csv', delimiter=',',skip_header=1)
      glove_emb_q2 = genfromtxt('/tmp/Ganesh_MSCI/Unbalanced_Embeddings/glove/glove_q2_unbalanced.csv', delimiter=',',skip_header=1)
      glove_emb_q1 = np.delete(glove_emb_q1,0, 1)
      glove_emb_q2 = np.delete(glove_emb_q2, 0, 1)
      print('Loading Embeddings BERT')
      bert_q = genfromtxt('/tmp/Ganesh_MSCI/Unbalanced_Embeddings/bert/bert_qpair_unbalanced.csv', delimiter=',',skip_header=1)
      bert_q = np.delete(bert_q,0,1)
      bert_q1 = genfromtxt('/tmp/Ganesh_MSCI/Unbalanced_Embeddings/bert/bert_q1_unbalanced.csv', delimiter=',',skip_header=1)
      bert_q1 = np.delete(bert_q1,0,1)
      print('Loading Embeddings BERTQ2')
      bert_q2 = genfromtxt('/tmp/Ganesh_MSCI/Unbalanced_Embeddings/bert/bert_q2_unbalanced.csv', delimiter=',',skip_header=1)
      bert_q2 = np.delete(bert_q2,0,1)
      
    elif sys.argv[1] == "b":
      print('Loading Embeddings W2vec')
      w2v_emb_q1 = genfromtxt('/tmp/Ganesh_MSCI/balanced_Embeddings/word2vec/w2vec_q1_balanced.csv', delimiter=',',skip_header=1)
      w2v_emb_q2 = genfromtxt('/tmp/Ganesh_MSCI/balanced_Embeddings/word2vec/w2vec_q2_balanced.csv', delimiter=',',skip_header=1)

      print('Loading Embeddings fastext')
      ft_emb_q1 = genfromtxt('/tmp/Ganesh_MSCI/balanced_Embeddings/fastext/fastext_q1_balanced.csv', delimiter=',',skip_header=1)
      ft_emb_q2 = genfromtxt('/tmp/Ganesh_MSCI/balanced_Embeddings/fastext/fastext_q2_balanced.csv', delimiter=',',skip_header=1)
      
      print('Loading Embeddings glove')
      glove_emb_q1 = genfromtxt('/tmp/Ganesh_MSCI/balanced_Embeddings/glove/glove_q1_balanced.csv', delimiter=',',skip_header=1)
      glove_emb_q2 = genfromtxt('/tmp/Ganesh_MSCI/balanced_Embeddings/glove/glove_q2_balanced.csv', delimiter=',',skip_header=1)
      w2v_emb_q1 = np.delete(w2v_emb_q1, 0, 1)
      w2v_emb_q2 = np.delete(w2v_emb_q2, 0, 1)

      ft_emb_q1 = np.delete(ft_emb_q1, 0, 1)
      ft_emb_q2 = np.delete(ft_emb_q2, 0, 1)

      glove_emb_q1 = np.delete(glove_emb_q1, 0, 1)
      glove_emb_q2 = np.delete(glove_emb_q2, 0, 1)
      print('Loading Embeddings BERT')
      bert_q = genfromtxt('/tmp/Ganesh_MSCI/balanced_Embeddings/bert/bert_qpair_balanced.csv', delimiter=',',skip_header=1)
      bert_q = np.delete(bert_q,0,1)
      print('Loading Embeddings BERTQ1')
      bert_q1 = genfromtxt('/tmp/Ganesh_MSCI/balanced_Embeddings/bert/bert_q1_balanced.csv', delimiter=',',skip_header=1)
      bert_q1 = np.delete(bert_q1,0,1)
      print('Loading Embeddings BERTQ2')
      bert_q2 = genfromtxt('/tmp/Ganesh_MSCI/balanced_Embeddings/bert/bert_q2_balanced.csv', delimiter=',',skip_header=1)
      bert_q2 = np.delete(bert_q2,0,1)

  # print("Getting Bert Embeddings..")
  # bert_e = get_bertembeddings(q1sents,q2sents)
  # print('Bert Embeddings Shape',bert_e.shape)
  #Preparing Data for Training Network
  # df_sub = df_sub.reindex(np.random.permutation(df_sub.index))
  features = add_features()
  # set number of train and test instances

  num_train = int(df_sub.shape[0] * 0.70)
  num_val = int(df_sub.shape[0] * 0.10)
  num_test = df_sub.shape[0] - num_train - num_val 

  # total = np.arange(df_sub.shape[0])
  # num_train1,num_test = train_test_split(total,test_size=0.1,random_state=33)
  # num_train,num_val = train_test_split(num_train1,test_size=1/9)
              
  print("Number of training pairs: %i"%(num_train))
  print("Number of Validation pairs: %i"%(num_val))
  print("Number of testing pairs: %i"%(num_test))


    # init data data arrays
  X_train_cnn_a = np.zeros([num_train, 324, 3])
  X_test_cnn_a  = np.zeros([num_test, 324, 3])
  X_val_cnn_a  = np.zeros([num_val, 324, 3])

  X_train_cnn_b = np.zeros([num_train, 324, 3])
  X_test_cnn_b  = np.zeros([num_test, 324, 3])
  X_val_cnn_b  = np.zeros([num_val, 324, 3])

  Y_train = np.zeros([num_train]) 
  Y_test = np.zeros([num_test])
  Y_val = np.zeros([num_val]) 


  #Labels
  Y_train = df_sub['is_duplicate'].values[num_train]
  Y_val = df_sub['is_duplicate'].values[num_val]
  Y_test = df_sub['is_duplicate'].values[num_val]
  


  num_val = num_train + int(df_sub.shape[0] * 0.10)
  # fill data arrays with features
  X_train_cnn_a[:,:,0] = ft_emb_q1[:num_train]
  X_train_cnn_a[:,:,1] = w2v_emb_q1[:num_train]
  X_train_cnn_a[:,:,2] = glove_emb_q1[:num_train]

  X_train_cnn_b[:,:,0] = ft_emb_q2[:num_train]
  X_train_cnn_b[:,:,1] = w2v_emb_q2[:num_train]
  X_train_cnn_b[:,:,2] = glove_emb_q2[:num_train]

  features_train = features[:num_train]
  features_b_train = bert_q[:num_train]
  Y_train = df_sub[:num_train]['is_duplicate'].values

  X_val_cnn_a[:,:,0] = ft_emb_q1[num_train:num_val]
  X_val_cnn_a[:,:,1] = w2v_emb_q1[num_train:num_val]
  X_val_cnn_a[:,:,2] = glove_emb_q1[num_train:num_val]

  X_val_cnn_b[:,:,0] = ft_emb_q2[num_train:num_val]
  X_val_cnn_b[:,:,1] = w2v_emb_q2[num_train:num_val]
  X_val_cnn_b[:,:,2] = glove_emb_q2[num_train:num_val]

  features_val = features[num_train:num_val]
  features_b_val = bert_q[num_train:num_val]
  Y_val = df_sub[num_train:num_val]['is_duplicate'].values


  X_test_cnn_a[:,:,0] = ft_emb_q1[num_val:]
  X_test_cnn_a[:,:,1] = w2v_emb_q1[num_val:]
  X_test_cnn_a[:,:,2] = glove_emb_q1[num_val:]

  X_test_cnn_b[:,:,0] = ft_emb_q2[num_val:]
  X_test_cnn_b[:,:,1] = w2v_emb_q2[num_val:]
  X_test_cnn_b[:,:,2] = glove_emb_q2[num_val:]
  features_test = features[num_val:]
  features_b_test = bert_q[num_val:]
  Y_test = df_sub[num_val:]['is_duplicate'].values

  


 


 
  
  
  #############Pipeline Starts################
  net = create_network([324,3],25)
  optimizer = Adam(lr=0.001)
  net.compile(loss="binary_crossentropy", optimizer=optimizer,metrics=['accuracy'])
  print(net.summary())


  # # Training Sets for LSTM1

  X_intera_train_1 = X_train_cnn_a[:,:,0]
  X_interb_train_1 = X_train_cnn_b[:,:,0]
  X_train_lstm1_a = X_intera_train_1[:,:,np.newaxis]
  X_train_lstm1_b = X_interb_train_1[:,:,np.newaxis]

  X_intera_val_1 = X_val_cnn_a[:,:,0]
  X_interb_val_1 = X_val_cnn_b[:,:,0]
  X_val_lstm1_a = X_intera_val_1[:,:,np.newaxis]
  X_val_lstm1_b = X_interb_val_1[:,:,np.newaxis]

  X_intera_test_1 = X_test_cnn_a[:,:,0]
  X_interb_test_1 = X_test_cnn_b[:,:,0]
  X_test_lstm1_a = X_intera_test_1[:,:,np.newaxis]
  X_test_lstm1_b = X_interb_test_1[:,:,np.newaxis]


  # Validation Sets for LSTM2

  X_intera_train_2 = X_train_cnn_a[:,:,1]
  X_interb_train_2 = X_train_cnn_b[:,:,1]
  X_train_lstm2_a = X_intera_train_2[:,:,np.newaxis]
  X_train_lstm2_b = X_interb_train_2[:,:,np.newaxis]

  X_intera_val_2 = X_val_cnn_a[:,:,1]
  X_interb_val_2 = X_val_cnn_b[:,:,1]
  X_val_lstm2_a = X_intera_val_2[:,:,np.newaxis]
  X_val_lstm2_b = X_interb_val_2[:,:,np.newaxis]

  X_intera_test_2 = X_test_cnn_a[:,:,1]
  X_interb_test_2 = X_test_cnn_b[:,:,1]
  X_test_lstm2_a = X_intera_test_2[:,:,np.newaxis]
  X_test_lstm2_b = X_interb_test_2[:,:,np.newaxis]
  
  # Test Set for LSTM3

  X_intera_train_3 = X_train_cnn_a[:,:,2]
  X_interb_train_3 = X_train_cnn_b[:,:,2]
  X_train_lstm3_a = X_intera_train_3[:,:,np.newaxis]
  X_train_lstm3_b = X_interb_train_3[:,:,np.newaxis]
 
  X_intera_val_3 = X_val_cnn_a[:,:,2]
  X_interb_val_3 = X_val_cnn_b[:,:,2]
  X_val_lstm3_a = X_intera_val_3[:,:,np.newaxis]
  X_val_lstm3_b = X_interb_val_3[:,:,np.newaxis]
  
  X_intera_test_3 = X_test_cnn_a[:,:,2]
  X_interb_test_3 = X_test_cnn_b[:,:,2]
  X_test_lstm3_a = X_intera_test_3[:,:,np.newaxis]
  X_test_lstm3_b = X_interb_test_3[:,:,np.newaxis]


  # Test Set for LSTM4 BERT

  X_intera_train_4 = bert_q1[:num_train]
  X_train_lstm4_a = X_intera_train_4[:,:,np.newaxis]

  X_intera_val_4 = bert_q1[num_train:num_val]
  X_val_lstm4_a = X_intera_val_4[:,:,np.newaxis]

  X_intera_test_4 = bert_q1[num_val:]
  X_test_lstm4_a = X_intera_test_4[:,:,np.newaxis]

  X_interb_train_4 = bert_q2[:num_train]
  X_train_lstm4_b = X_interb_train_4[:,:,np.newaxis]

  X_interb_val_4 = bert_q2[num_train:num_val]
  X_val_lstm4_b = X_interb_val_4[:,:,np.newaxis]

  X_interb_test_4 = bert_q2[num_val:]
  X_test_lstm4_b = X_interb_test_4[:,:,np.newaxis]

  print("Input Shapes")
  print("CNN Shape")
  print(X_train_cnn_a.shape,X_val_cnn_a.shape,X_test_cnn_a.shape)
  print("LSTM (x3) Shape:")
  print(X_train_lstm4_a.shape,X_val_lstm4_a.shape,X_test_lstm4_a.shape)

  print("Features shape:",features_train.shape,features_val.shape,features_test.shape)
  print("BERT Features shape:",features_b_train.shape,features_b_val.shape,features_b_test.shape)
  
  print("Labels Shape")
  print(Y_train.shape,Y_val.shape,Y_test.shape)

  filepath="./QQP_{epoch:02d}_{val_loss:.4f}.h5"
  checkpoint = callbacks.ModelCheckpoint(filepath, 
                                        monitor='val_loss', 
                                        verbose=0, 
                                        save_best_only=True)
  callbacks_list = [checkpoint]
  
  if sys.argv[3] == "train":
    for epoch in range(1):
      if sys.argv[4] == "resume":
        print('Resuming Model..')
        net = load_model(sys.argv[5])
        #Add new config to trained model



      # net.fit([X_train_cnn_a, X_train_cnn_b, X_train_lstm4_a, X_train_lstm4_b,features_train,features_b_train], 
      #           Y_train,
      #         validation_data=([X_val_cnn_a, X_val_cnn_b,X_val_lstm4_a, X_val_lstm4_b,features_val,features_b_val]
      #                         , Y_val),
      #         batch_size=384, nb_epoch=1, shuffle=True,callbacks = callbacks_list)

      net.fit([ X_train_cnn_a, X_train_cnn_b,X_train_lstm1_a, X_train_lstm1_b,
                X_train_lstm2_a, X_train_lstm2_b,X_train_lstm3_a, X_train_lstm3_b,X_train_lstm4_a, X_train_lstm4_b,features_train,features_b_train], 
                Y_train,
              validation_data=([X_val_cnn_a, X_val_cnn_b,X_val_lstm1_a, X_val_lstm1_b,
                              X_val_lstm2_a, X_val_lstm2_b,X_val_lstm3_a, X_val_lstm3_b,X_val_lstm4_a, X_val_lstm4_b,features_val,features_b_val]
                              , Y_val),
              batch_size=384, nb_epoch=16, shuffle=True)

      # net.fit([X_train_cnn_a, X_train_cnn_b, X_train_lstm1_a, X_train_lstm1_b,
      #           X_train_lstm2_a, X_train_lstm2_b,X_train_lstm3_a, X_train_lstm3_b,features_train,features_b_train], 
      #           Y_train,
      #         validation_data=([X_val_cnn_a, X_val_cnn_b,X_val_lstm1_a, X_val_lstm1_b,
      #                         X_val_lstm2_a, X_val_lstm2_b,X_val_lstm3_a, X_val_lstm3_b,features_val,features_b_val]
      #                         , Y_val),
      #         batch_size=384, nb_epoch=16, shuffle=True)
    # score = net.evaluate([X_test_cnn_a, X_test_cnn_b,X_test_lstm4_a, X_test_lstm4_b,features_test,features_b_test],Y_test,batch_size=384)
    score = net.evaluate([X_test_cnn_a, X_test_cnn_b,X_test_lstm1_a, X_test_lstm1_b,
                  X_test_lstm2_a, X_test_lstm2_b,X_test_lstm3_a, X_test_lstm3_b,X_test_lstm4_a, X_test_lstm4_b,features_test,features_b_test],Y_test,batch_size=384)
    print('Test loss : {:.4f}'.format(score[0]))
    print('Test accuracy : {:.4f}'.format(score[1]))
  else:
    net = load_model('QQP_07_0.6652.h5')
    score = net.evaluate([X_test_cnn_a, X_test_cnn_b,X_test_lstm1_a, X_test_lstm1_b,
                  X_test_lstm2_a, X_test_lstm2_b,X_test_lstm3_a, X_test_lstm3_b,features_test,features_b_test],Y_test,batch_size=384)
    print('Test loss : {:.4f}'.format(score[0]))
    print('Test accuracy : {:.4f}'.format(score[1]))
  return 0


if __name__== "__main__":
  main()





