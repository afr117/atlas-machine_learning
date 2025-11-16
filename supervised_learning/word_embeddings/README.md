# Word Embeddings

This project focuses on Natural Language Processing (NLP) techniques for converting text into numerical vector representations (embeddings). These embeddings allow machine learning models to process and understand language data.

---

## 📌 Learning Objectives

By completing this project, you should be able to explain:

- What **natural language processing (NLP)** is
- What **word embedding** represents
- What **Bag of Words (BoW)** is
- What **TF-IDF** is and why it improves over BoW
- What **CBOW** (Continuous Bag of Words) is
- What **Skip-gram** is and how it differs from CBOW
- What **Word2Vec**, **GloVe**, **fastText**, and **ELMo** are
- What **negative sampling** is
- What **n-grams** are

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `0-bag_of_words.py` | Creates a Bag of Words embedding matrix |
| `1-tf_idf.py` | Computes TF-IDF embedding matrix |
| `2-word2vec.py` | Trains a Word2Vec model using gensim |
| `3-gensim_to_keras.py` | Converts a gensim model to a Keras Embedding layer |
| `4-fasttext.py` | Trains a fastText model using gensim |
| `5-elmo` | Contains the written multiple-choice answer for ELMo |

---

## 🧪 Dependencies

- Python 3.9
- numpy 1.25.2
- tensorflow 2.15
- gensim 4.3.3

Install gensim (as required):
```bash
pip install --user gensim==4.3.3
Check TensorFlow/Keras version:

python
Copy code
import keras; print(keras.__version__)
# Expected: 2.15.0
🧠 Summary of Methods
Method	What It Captures	Pros	Cons
BoW	Word frequency	Simple	Ignores meaning & order
TF-IDF	Word importance across documents	More semantic relevance	Still ignores order
Word2Vec	Context-based embeddings	Semantic relationships	Requires training
fastText	Subword information	Handles rare/unknown words	Slower training
ELMo	Contextual + deep embeddings	Strong performance	Computationally heavy

📝 Requirements (as enforced by checker)
Files are executable

First line: #!/usr/bin/env python3

Code follows pycodestyle (2.11.1)

All modules, classes, and functions include documented docstrings

No external NLP libraries besides gensim and tensorflow

👤 Author
Alfredo Figueroa
Student — Atlas School
GitHub: https://github.com/afr117
