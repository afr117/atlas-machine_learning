This project focuses on implementing common evaluation metrics used in Natural Language Processing (NLP) to compare model-generated text with human-written references. You will calculate BLEU scores (unigram, n-gram, and cumulative BLEU) without relying on external NLP libraries such as NLTK.

---

## 📌 Learning Objectives

By completing this project, you should be able to explain:

- What **Natural Language Processing (NLP)** is used for
- What a **BLEU score** measures and how it is computed
- What a **ROUGE score** measures
- What **perplexity** represents in language modeling
- When to choose one evaluation metric over another

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `0-uni_bleu.py` | Computes unigram BLEU score |
| `1-ngram_bleu.py` | Computes BLEU using a specific n-gram size |
| `2-cumulative_bleu.py` | Computes cumulative BLEU score from 1 up to n-grams |

Each module contains complete documentation and passes the checker’s required test cases.

---

## 🧪 Example Usage

```bash
./0-main.py   # Unigram BLEU
./1-main.py   # N-gram BLEU (e.g., bigrams)
./2-main.py   # Cumulative BLEU (1→N)
Example output:

text
Copy code
0.6549846024623855
0.6140480648084865
0.5475182535069453
🔍 Summary of Metrics
Metric	Focus	Strength	Limitation
BLEU	Precision of overlapping n-grams	Stable for MT	Struggles with synonyms & meaning
ROUGE	Recall of reference text	Great for summarization	Not implemented in this project
Perplexity	Predictive uncertainty	Common in LM training	Hard for comparing different vocabularies

When choosing a metric:

MT → BLEU

Summarization → ROUGE

Language Modeling → Perplexity

📝 Requirements
Python 3.9

numpy 1.25.2

All files executable

First line of all scripts:
#!/usr/bin/env python3

Code follows pycodestyle 2.11.1

No use of NLTK or other external NLP libraries

👤 Author
Alfredo Figueroa
Machine Learning Student — Atlas School
GitHub: https://github.com/afr117
