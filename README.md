# Sherlock Holmes N-Gram Predictor

A statistical language model built from scratch in Python to predict the next word in a sentence using patterns from Sir Arthur Conan Doyle's literature.

## 📖 Project Summary
This project implements a word-level **N-Gram Language Model** (defaulting to 4-grams). It processes raw text from the Sherlock Holmes canon, builds frequency-based probability tables, and uses an inference engine to guess the most likely following word based on user input.

### 🧠 Technical Logic
* **Architecture:** Uses a 4-level dictionary structure (1-gram through 4-gram) to store context and word counts.
* **Probability:** Implements **Maximum Likelihood Estimation (MLE)** to convert raw counts into decimal probabilities.
* **Smoothing:** Uses **Stupid Backoff** logic. If the model hasn't seen a specific 4-word sequence, it "backs off" to a 3-word sequence, and so on, until a match is found.
* **OOV Handling:** Replaces rare words (appearing fewer than 3 times) with an `<UNK>` (Unknown) token to improve model generalization.


---

## 📂 Project Structure
```text
ngram-predictor/
├── config/              # Environment variables (.env)
├── data/                # Raw books, processed tokens, and JSON models
├── src/
│   ├── data_prep/       # Normalizer: Cleans and tokenizes text
│   ├── model/           # NGramModel: Training and Probability logic
│   └── inference/       # Predictor: The UI/Inference wrapper
├── main.py              # Central entry point
└── requirements.txt     # Dependencies