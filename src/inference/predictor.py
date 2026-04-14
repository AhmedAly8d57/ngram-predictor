import logging
import os
import sys

# Ensure the root directory is in the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, model, normalizer):
        """
        Requirement 1: Accept pre-loaded NGramModel and Normalizer instances.
        """
        self.model = model
        self.normalizer = normalizer
        logger.info("Predictor initialized with pre-loaded model and normalizer.")

    def normalize(self, text):
        """
        Requirement 2: Call Normalizer.normalize(text) and extract context.
        """
        cleaned_text = self.normalizer.normalize(text)
        tokens = self.normalizer.word_tokenize(cleaned_text)
        
        # NGRAM_ORDER is 4, so context is the last 3 words
        context = tokens[-3:] if tokens else []
        return context

    def map_oov(self, context):
        """
        Requirement 3: Replace out-of-vocabulary words with <UNK>.
        """
        mapped_context = [
            word if word in self.model.vocab else "<UNK>" 
            for word in context
        ]
        return tuple(mapped_context)

    def predict_next(self, text, k=3):
        """
        Requirement 4: Orchestrate normalize -> map_oov -> lookup.
        """
        raw_context = self.normalize(text)
        safe_context = self.map_oov(raw_context)
        
        probs_dict = self.model.lookup(safe_context)
        
        if not probs_dict:
            return []

        # Sort by probability descending
        sorted_predictions = sorted(
            probs_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [word for word, prob in sorted_predictions[:k]]
