import re
import string

class Normalizer:
    """
    Cleans and standardizes text data for the N-Gram model.
    This is a dual-use class used for both training data and user input.
    """

    def __init__(self):
        """Initializes the Normalizer."""
        pass

    def normalize(self, text: str) -> str:
        """
        Applies the full normalization pipeline to a text string.
        1. Lowercases text
        2. Removes punctuation
        3. Removes numbers
        4. Removes extra whitespace
        """
        # Step 1: Lowercase
        text = text.lower()
        
        # Step 2: Remove Punctuation (Replaces all punctuation with 'nothing')
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Step 3: Remove Numbers (The '\d+' means "find any digit")
        text = re.sub(r'\d+', '', text)
        
        # Step 4: Remove extra whitespace (Turns double spaces into single spaces)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text