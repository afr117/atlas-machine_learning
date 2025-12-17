# Question Answering Chatbot Project

This project focuses on building a full-pipeline Question-Answering (QA) chatbot using Natural Language Processing (NLP). It combines Semantic Search to find relevant documents and BERT-based models to extract precise answers from text.

## Learning Objectives
By the end of this project, I can explain:
* **Question-Answering (QA):** The process of automatically extracting a specific answer from a given context based on a natural language query.
* **Semantic Search:** Searching by meaning rather than exact keyword matching, using vector embeddings to find document similarity.
* **BERT (Bidirectional Encoder Representations from Transformers):** A deeply bidirectional model that understands the context of a word based on its surroundings (both left and right).
* **Transformers Library:** A tool by Hugging Face used to implement pre-trained NLP models like BERT.
* **TensorFlow Hub:** A repository of trained machine learning models used here for the Universal Sentence Encoder.

---

## Tasks Summary

### 0. Question Answering
Implemented a function that uses a pre-trained BERT model (`bert-large-uncased-whole-word-masking-finetuned-squad`) to find the start and end indices of an answer within a reference document.

### 1. Create the Loop
Developed an interactive command-line interface that prompts the user for input (`Q:`) and handles case-insensitive exit commands like `bye`, `exit`, or `quit`.

### 2. Answer Questions
Combined the interactive loop with the BERT QA model to create a bot that can answer questions based on a single provided document. It includes a fallback response if the answer is not found.

### 3. Semantic Search
Created a search engine that scans a directory of Markdown files. It uses the Universal Sentence Encoder to convert documents into vectors and uses cosine similarity to find the document most relevant to a user's question.

### 4. Multi-reference Question Answering
The final chatbot integration. This bot takes a user's question, performs a semantic search across a corpus of documents to find the best reference, and then uses the BERT model to extract the specific answer from that document.

---

## Requirements
* **OS:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Libraries:** * `tensorflow` (2.15)
  * `tensorflow-hub` (0.15.0)
  * `transformers` (4.44.2)
  * `numpy` (1.25.2)
* **Style:** `pycodestyle` (version 2.11.1)

## How to Use
Run the main chatbot:
```bash
./4-main.py
Ask questions about school policies or PLDs, and type bye to exit.


---

### **Final Pro-Tip for Submission**
Before you push your code to GitHub, ensure all your files have the proper permissions. Since you are working on Windows, the "executable" bit might not be set. When you are on your Ubuntu terminal/checker, run:

```bash
chmod +x 0-qa.py 1-loop.py 2-qa.py 3-semantic_search.py 4-qa.py
