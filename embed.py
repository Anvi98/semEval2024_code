"""This script is extract embeddings of sentences from SentenceBert and compute the cosine similarity of each pair of sentence."""

import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

raw = pd.read_csv("eng_train.csv")
raw2 = pd.read_csv("eng_dev_with_labels.csv")
raw3 = pd.read_csv("eng_test.csv")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

sents = list(raw2["Text"])

sent_pair = []

for sent in sents:
    sentences = sent.split("\n", 1)
    sent_pair.append([sentences[0], sentences[1]])

results = []

i = 0
for pair in sent_pair:
     tmp_emb1 = model.encode(pair[0], convert_to_tensor=True)
     tmp_emb2 = model.encode(pair[1], convert_to_tensor=True)
     tmp_cos = util.cos_sim(np.array(tmp_emb1.cpu()), np.array(tmp_emb2.cpu())).numpy()[0][0]
     results.append(tmp_cos)
     i +=1
     print(i)
     #print(tmp_cos)

results = np.array(results)
np.savez_compressed("pap_res_vo_dev_lab", results)
