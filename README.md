# semEval2024: Semantic Textual Relatedness - Pinealai Code 
## Description Task 1 English
A shared task on automatically detecting the degree of semantic relatedness between pairs of sentences. New textual datasets will be provided for Afrikaans, Algerian Arabic, Amharic, English, Hausa, Hindi, Indonesian, Kinyarwanda, Marathi, Moroccan Arabic, Modern Standard Arabic, Punjabi, Spanish, and Telugu. 

## Main objective
The main objective for us, was to identify and find features and explainable way of the impact of those features in predicting semantic textual similarity in English.

## Brief Methodology
In this study, our objective is to predict semantic textual relatedness between two texts. We made
two key assumptions:
- We refrained from preprocessing the corpus to preserve sentence structure, essential for information retrieval and semantic identification (Hirst, 1987);
- We intentionally excluded Large Language Models (LLMs) from experiments, anticipating challenges in interpreting specific features contributing to semantic identification due to their contextual abilities and complexity.

Below is the diagram of our method:

![Diagram of Pinealai Methodology](Method-STR.png "Pinealai Methodology")

## How to reproduce ?
### Requirements
To be able to reproduce the work, you will need to create a virtual environment and install the dependencies we  used. The main ones are: scikit-learn, spacy, nltk and sentence_transformers. But it is better, you install the dependencies from the requirements.txt in the folder. For the jupyer notebook code, it will be easy to know the dependencies needed.

So, first thing is to clone the repo, and set it locally:
``` 
git clone https://github.com/Anvi98/semEval2024_code.github 
```
Create the virtual environment and activate it. (you need to be located in the folder):
``` 
python3 -m venv env 
source env/bin/activate
``` 
Then, install the dependencies. (simply 'pip' if not using python3):
``` 
pip3 install -r requirements.txt
``` 
After installing, the dependencies, everything is set. You will be able to run these files:
- embed.py (Extract Bert embedding)
- fuzzy_h.py (Computed Levenshtein distance of pair of sentences)
- syntactic_features.py (Extract syntactic features of pair of sentences)
- synt.py (Training and prediction of traditional ML models)


## Authors:
- Anvi Alex Eponon
- Luis Ramos
