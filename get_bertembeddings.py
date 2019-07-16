import sys
import os
import numpy as np_array
import pandas as pd
from bert_serving.client import BertClient

def get_bertembeddings_pair(q1,q2):
  q = list(map(lambda x, y: x+ ' ||| ' +y, q1, q2))
  bc = BertClient()
  bert_embeddings = bc.encode(q)
  return bert_embeddings

def get_bertembeddings_sent(q):
  bc = BertClient()
  bert_embeddings = bc.encode(q)
  return bert_embeddings

def main():
  #Code for Balanced Data
  df_sub = pd.read_csv('data_balanced.csv')
  
  print('Shape of Dataset',df_sub.shape)

  df_sub['question1'] = df_sub['question1'].apply(lambda x: str(x))
  df_sub['question2'] = df_sub['question2'].apply(lambda x: str(x))
  q1sents_b = list(df_sub['question1'])
  q2sents_b = list(df_sub['question2'])

  print("Getting Bert Embeddings pairs-balanced..")
  bert_b = get_bertembeddings_pair(q1sents_b,q2sents_b)
  pd.DataFrame(bert_b).to_csv("bert_qpair_balanced.csv")

  print("Getting Bert Embeddings Q1-balanced..")
  bert_b_q1 = get_bertembeddings_sent(q1sents_b)
  pd.DataFrame(bert_b_q1).to_csv("bert_q1_balanced.csv")

  print("Getting Bert Embeddings Q2-balanced..")
  bert_b_q2 = get_bertembeddings_sent(q2sents_b)
  pd.DataFrame(bert_b_q2).to_csv("bert_q2_balanced.csv")


  ############Code for Unbalanced Data
  df_sub_ubal = pd.read_csv('data_unbalanced.csv')
  
  print('Shape of Dataset-unbalanced',df_sub.shape)
  df_sub_ubal['question1'] = df_sub_ubal['question1'].apply(lambda x: str(x))
  df_sub_ubal['question2'] = df_sub_ubal['question2'].apply(lambda x: str(x))
  q1sents_u = list(df_sub_ubal['question1'])
  q2sents_u = list(df_sub_ubal['question2'])

  print("Getting Bert Embeddings pairs-unbalanced..")
  bert_u = get_bertembeddings_pair(q1sents_u,q2sents_u)
  pd.DataFrame(bert_u).to_csv("bert_qpair_balanced.csv")

  print("Getting Bert Embeddings Q1-unbalanced..")
  bert_u_q1 = get_bertembeddings_sent(q1sents_u)
  pd.DataFrame(bert_b_q1).to_csv("bert_q1_balanced.csv")

  print("Getting Bert Embeddings Q2-unbalanced..")
  bert_u_q2 = get_bertembeddings_sent(q2sents_u)
  pd.DataFrame(bert_u_q2).to_csv("bert_q2_balanced.csv")