import os
import json
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class NGramModel:
    def __init__(self):
        # The set of words that meet our frequency threshold
        self.vocab = set()
        
        # Nested counts: {n: {context: {next_word: count}}}
        # 1-gram is just a simple Counter: {word: count}
        self.counts = {
            1: Counter(),
            2: defaultdict(Counter),
            3: defaultdict(Counter),
            4: defaultdict(Counter)
        }

    def build_vocab(self, token_filepath, threshold=3):
        """Requirement 1: Filter rare words and identify the 'known' world."""
        all_words = []
        with open(token_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                all_words.extend(line.split())
        #This line create a dict that contain every element and its count
        word_counts = Counter(all_words)
        
        # Set comprehension: Keep only words appearing >= threshold
        self.vocab = {word for word, count in word_counts.items() if count >= threshold}
        
        # Requirement 2: Always include special control tokens
        self.vocab.add("<UNK>")
        self.vocab.add("<s>")
        self.vocab.add("</s>")
        
        logger.info(f"Vocabulary built with threshold {threshold}. Size: {len(self.vocab)}")

    def build_counts_and_probabilities(self, token_file):
            """
            Requirement 2: Count n-grams (1-4) and compute MLE probabilities.
            """
            logger.info("Counting n-grams and computing probabilities...")

            # --- STEP 1: COUNTING ---
            with open(token_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # We save that list into the variable 'tokens'.
                    tokens = line.split()
                    if not tokens: continue

                    # We apply the 'Padding' and 'UNK' logic directly here 
                    # to keep the sentence ready for a 4-gram model.
                    # STEP B: Now we create a new empty list called 'padded'.
                    padded = ["<s>"] * 3
                    # STEP C: This is where 'padded' meets 'tokens'!
                    # The loop below looks at every word INSIDE 'tokens' 
                    # and copies it into 'padded'.
                    for t in tokens:
                        padded.append(t if t in self.vocab else "<UNK>")
                    # STEP D: Final touch
                    # add </s> to know that the line end
                    padded.append("</s>")

                    # The Sliding Window
                    for i in range(len(padded)):
                        word = padded[i]
                        
                        # Unigram (Order 1)
                        self.counts[1][word] += 1
                        
                        # Bigram (Order 2)
                        if i >= 1:
                            self.counts[2][padded[i-1]][word] += 1
                        
                        # Trigram (Order 3)
                        if i >= 2:
                            self.counts[3][(padded[i-2], padded[i-1])][word] += 1
                        
                        # Quadgram (Order 4)
                        if i >= 3:
                            self.counts[4][(padded[i-3], padded[i-2], padded[i-1])][word] += 1

            # --- STEP 2: PROBABILITIES (MLE) ---
            self.probs = {1: {}, 2: {}, 3: {}, 4: {}}
            
            # Calculate Order 1 (Unigram) - No context, divide by total
            total_words = sum(self.counts[1].values())
            for word, count in self.counts[1].items():
                self.probs[1][word] = count / total_words

            # Calculate Orders 2, 3, 4 - Divide by context total
            for n in [2, 3, 4]:
                for context, next_words in self.counts[n].items():
                    self.probs[n][context] = {}
                    context_total = sum(next_words.values())
                    for word, count in next_words.items():
                        self.probs[n][context][word] = count / context_total
# --- UNIT TESTER ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import tempfile
    
    # 1. Setup a fake data file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
        # 'ahmed' is frequent (3), 'sherlock' is rare (1)
        tmp.write("ahmed is here\nahmed is there\nahmed is great\nsherlock is gone")
        tmp_path = tmp.name

    try:
        model = NGramModel()
        
        # 2. Test Vocabulary
        model.build_vocab(tmp_path, threshold=2)
        print(f"Is 'ahmed' in vocab? { 'ahmed' in model.vocab }") # True
        print(f"Is 'sherlock' in vocab? { 'sherlock' in model.vocab }") # False
        
        # 3. Test Training Counts
        model.train(tmp_path)
        
        # Check Unigram
        print(f"Unigram count for 'ahmed': {model.counts[1]['ahmed']}") # Should be 3
        
        # Check Bigram (context <s> -> word ahmed)
        # In '<s> <s> <s> ahmed...', the sequence (<s> -> ahmed) happens once per sentence
        print(f"Bigram count for '<s>' -> 'ahmed': {model.counts[2]['<s>']['ahmed']}")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)