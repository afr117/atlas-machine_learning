Word Embeddings Project



This project explores Natural Language Processing (NLP) with a focus on word embeddings, ranging from simple techniques like Bag of Words and TF-IDF, to distributed vector representations such as Word2Vec, GloVe, FastText, and ELMo.



The primary goal is to learn how computers represent and understand human language and implement foundational embedding techniques using Python.



📌 Learning Objectives



By the end of this project, you should be able to explain the following concepts clearly:



🔹 General NLP Concepts



What is natural language processing?



What is a word embedding?



What is Bag of Words (BoW)?



What is TF-IDF?



What is CBOW (Continuous Bag of Words)?



What is a Skip-Gram?



What is an n-gram?



What is negative sampling?



What are Word2Vec, GloVe, FastText, and ELMo?



🛠️ Technical Requirements



Editors Allowed: vi, vim, emacs



OS \& Python: Ubuntu 20.04 LTS · Python 3.9



Libraries:



numpy 1.25.2



tensorflow 2.15



gensim 4.3.3 (when allowed by tasks)



Each file must:



end with a new line



start with the line:



\#!/usr/bin/env python3





be executable



follow pycodestyle rules (version 2.11.1)



All modules, functions, and classes must include proper documentation



📂 Repository Structure

atlas-machine\_learning/

└── supervised\_learning/

&nbsp;   └── word\_embeddings/

&nbsp;       ├── 0-bag\_of\_words.py

&nbsp;       ├── 1-tf\_idf.py

&nbsp;       ├── 2-word2vec.py

&nbsp;       ├── 3-get\_embs.py

&nbsp;       ├── 4-fasttext\_model.py

&nbsp;       ├── 5-elmo.py

&nbsp;       ├── ...

&nbsp;       ├── README.md

&nbsp;       └── ...





Note: Future task files will be added as required.



🧠 Task Summaries

Task	File	Description

0	0-bag\_of\_words.py	Implement Bag-of-Words embedding matrix (no Gensim allowed)

1	1-tf\_idf.py	Compute TF-IDF matrix

2	2-word2vec.py	Train Word2Vec model using Gensim

3	3-get\_embs.py	Retrieve Word2Vec embeddings

4	4-fasttext\_model.py	Build FastText word embeddings

5	5-elmo.py	Extract contextual embeddings using ELMo

…	…	Additional embeddings and evaluations

▶️ How to Run



Example for Task 0:



./0-main.py





Make sure scripts are executable:



chmod +x 0-main.py



📚 References



Mikolov et al. Word2Vec papers (2013)



Pennington et al. GloVe (2014)



Facebook AI Research — fastText



Deep Contextualized Word Representations — ELMo (2018)



👤 Author



Alfredo Figueroa

Atlas School — Machine Learning Track

GitHub: https://github.com/afr117

