
"""This script compute the levenshtein distance of each pair of sentences.
This will be part of the features for training the models."""

from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np

f = pd.read_csv("eng_train.csv")
f2 = pd.read_csv("eng_dev_with_labels.csv")
f3 = pd.read_csv("eng_test.csv")
texts = f2["Text"]

sents1 = []
sents2 = []

for sent in texts:
    sentences = sent.split("\n", 1)
    sents1.append(sentences[0])
    sents2.append(sentences[1])

res = []
for i in range(len(sents1)):
    
    similarity = fuzz.partial_ratio(sents1[i], sents2[i])
    res.append(similarity)
    with open("res_fuzz.txt", 'a+') as f2:
        f2.write(f"{similarity}\n")

res = np.array(res)
np.savez_compressed("pap_leven_score_dev_lab", res)
