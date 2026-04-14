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

    def lookup(self, context):
        """
        Requirement 3: Backoff lookup. 
        Try highest order first, fall back down to 1-gram.
        """
        # 1. Clean the context: Replace words not in vocab with <UNK>
        # We use a tuple because our dictionary keys (2, 3, 4-grams) are tuples
        clean_context = tuple(w if w in self.vocab else "<UNK>" for w in context)
        
        # 2. Determine the starting 'n' 
        # If context has 3 words, we look for a 4-gram (n=4)
        n = len(clean_context) + 1 

        # 3. THE BACKOFF LOOP (From Highest n down to 1)
        for order in range(n, 0, -1):
            if order == 1:
                # The "Safety Net": Always return the Unigram table
                return self.probs[1]
            
            # Get the correct "slice" of history for this order
            # e.g., for a 3-gram (order=3), we need the last 2 words (order-1)
            current_context = clean_context[-(order-1):]
            
            # BIGRAM SPECIAL CASE: 
            # In our build_counts, Bigram context was a single string, not a tuple.
            if order == 2:
                current_context = current_context[0]

            # Check if this "History" exists in our probability building
            if current_context in self.probs[order]:
                return self.probs[order][current_context]

        return {} # Fallback
    

    def save_model(self, model_path):
        """
        Requirement 4: Save all probability tables to model.json.
        """
        logger.info(f"Saving model probabilities to {model_path}...")
        
        # JSON cannot use Tuples as keys, so we convert ("a", "b") to "a b"
        serializable_probs = {1: self.probs[1], 2: {}, 3: {}, 4: {}}
        
        for n in [2, 3, 4]:
            for context, next_words in self.probs[n].items():
                # Join tuple ('my', 'dear') into string "my dear"
                key = " ".join(context) if isinstance(context, tuple) else context
                serializable_probs[n][key] = next_words
        
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_probs, f, indent=2)

    def save_vocab(self, vocab_path):
        """
        Requirement 5: Save vocabulary list to vocab.json.
        """
        logger.info(f"Saving vocabulary to {vocab_path}...")
        # Convert the Set to a sorted List so it looks nice in JSON
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(self.vocab)), f, indent=2)

    def load(self, model_path, vocab_path):
        """
        Requirement 6: Load model.json and vocab.json into the instance.
        """
        logger.info("Loading model and vocab from files...")
        
        # 1. Load Vocab
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = set(json.load(f))
            
        # 2. Load Probs
        with open(model_path, 'r', encoding='utf-8') as f:
            raw_probs = json.load(f)
            
            # Reconstruct the floors (converting "1" string keys back to int 1)
            self.probs = {1: raw_probs["1"], 2: {}, 3: {}, 4: {}}
            
            for n in ["2", "3", "4"]:
                int_n = int(n)
                for context_str, next_words in raw_probs[n].items():
                    # Convert "my dear" string back into tuple ("my", "dear")
                    key = tuple(context_str.split()) if int_n > 2 else context_str
                    self.probs[int_n][key] = next_words    
# --- UNIT TESTER ---
if __name__ == "__main__":
    # 1. Setup a fake model and fake vocab
    model = NGramModel()
    model.vocab = {"apple", "banana", "<s>", "</s>"} # Manual vocab for testing
    
    # 2. Create a tiny dummy file
    with open("test_tokens.txt", "w") as f:
        f.write("apple banana apple")
    
    # 3. Run the "Brain Building" method
    model.build_counts_and_probabilities("test_tokens.txt")
    
    # --- TEST THE COUNTS ---
    print("--- Floor 1 (Unigrams) ---")
    print(f"Count of 'apple': {model.counts[1]['apple']}") # Expected: 2
    
    print("\n--- Floor 2 (Bigrams) ---")
    # Context for bigram is just a string (the previous word)
    print(f"Count of 'banana' after 'apple': {model.counts[2]['apple']['banana']}") # Expected: 1
    
    # --- TEST THE PROBABILITIES ---
    print("\n--- MLE Probabilities ---")
    apple_prob = model.probs[1]['apple']
    # Total words = 3 (apple, banana, apple) + padding. 
    # Let's see what the model calculated:
    print(f"Probability of 'apple' (Order 1): {apple_prob:.4f}")
    
    # Test a 4-gram context (The start of the sentence)
    start_ctx = ("<s>", "<s>", "<s>")
    if start_ctx in model.probs[4]:
        print(f"Prob of 'apple' starting the sentence: {model.probs[4][start_ctx]['apple']}")