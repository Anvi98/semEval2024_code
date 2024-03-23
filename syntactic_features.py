"""This python script extracts the syntactic features from pair of sentences that will be later used by our models to learn the patterns that makes two sentences 
semantically related."""

from collections import Counter
from math import cos
import pandas as pd 
import numpy as np 
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_md")

raw = pd.read_csv("eng_train.csv")
raw_2 = pd.read_csv("eng_dev_with_labels.csv")
raw_3 = pd.read_csv("eng_test.csv")

# change the variable raw_2 to other files for feature extraction
texts = raw_2["Text"]
sample = raw_2.iloc[:, 1]

sents1 = []
sents2 = []

#Split Pair of sentences into two separate sentences
for sent in texts:
    sentences = sent.split("\n", 1)
    sents1.append(sentences[0])
    sents2.append(sentences[1])


data_texts = {"text1": sents1, "text2": sents2}
d_t = pd.DataFrame(data_texts)
d_t.to_csv("text_for_mistral_dev.csv")

#Parse the sentence with spacy
doc1 = [nlp(t.lower()) for t in sents1]
doc2 = [nlp(t.lower())for t in sents2]

# Extract dependency relations
dep_relations1 =[{token:token.dep_ for token in sent } for sent in doc1]
embed1 = [{token:token.vector for token in sent } for sent in doc1]

dep_relations2 = [{token:token.dep_ for token in sent } for sent in doc2]
embed2 = [{token:token.vector for token in sent } for sent in doc2]

# Create a set of unique dependency relations from both texts

# Create feature vectors
feature_vector1, feature_vector2 = [], []
total_sents = len(sents1)

for i in range(total_sents):
    unique_dep_relations = set(list(dep_relations1[i].values()) + list(dep_relations2[i].values()))
    tmp_features = []

    for dep in unique_dep_relations:
        token1 = [j for j in dep_relations1[i] if dep_relations1[i][j]==dep]
        token2 = [j for j in dep_relations2[i] if dep_relations2[i][j]==dep]
        if dep in dep_relations1[i].values() and dep in dep_relations2[i].values():
            if len(token1) == 1 and len(token2) == 1 and token1[0].orth_ == token2[0].orth_:
                pres = 1
                tmp_features.append((pres, 1)) 
            elif len(token1) == 1 and len(token2) == 1 and token1[0].orth_ != token2[0].orth_:
                emb1 = np.array(embed1[i][token1[0]]).reshape(1,-1)
                emb2 = np.array(embed2[i][token2[0]]).reshape(1,-1)
                cos_sim = cosine_similarity(emb1, emb2)[0][0] 
                pres = 0
                tmp_features.append((pres, cos_sim))
            elif len(token1) > 1 and len(token2) == 1:
                for l in range(len(token1)):
                     emb1 = np.array(embed1[i][token1[l]]).reshape(1,-1)
                     emb2 = np.array(embed2[i][token2[0]]).reshape(1,-1)
                     if token1[l].orth_ == token2[0].orth_:
                         pres = 1 
                     else:
                         pres = 0
                     cos_sim = cosine_similarity(emb1, emb2)[0][0] 

                     tmp_features.append((pres, cos_sim))
            elif len(token1) == 1 and len(token2) > 1:
                for l in range(len(token2)):
                     emb1 = np.array(embed1[i][token1[0]]).reshape(1,-1)
                     emb2 = np.array(embed2[i][token2[l]]).reshape(1,-1)
                     if token2[l].orth_ == token1[0].orth_:
                         pres = 1 
                     else:
                         pres = 0
                     cos_sim = cosine_similarity(emb1, emb2)[0][0] 

                     tmp_features.append((pres, cos_sim))
        else:
            tmp_features.append((0, 0)) 
    feature_vector1.append(tmp_features)

        
# Convert feature vectors to NumPy arrays
count = [] 
for vec in feature_vector1:
    num_1, num_0, num_R = 0,0,0 
    tmp = Counter(vec)
    sum_R = 0
    for k,v in tmp.items():
        if k[0] == 0 and k[1] > 0:
            sum_R += v
        if k[0] == 1:
            num_1 += v/len(vec)
        if k == (0,0):
            num_0 = v/len(vec)
    num_R = sum_R/len(vec)
    count.append([num_1,num_0,num_R])
    
feature_array1 = np.array(count)

#Save syntactic_features as a npz file
np.savez_compressed("pap_synctact_dev_lab", feature_array1)
