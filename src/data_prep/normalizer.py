import re
import os
import nltk

class Normalizer:
    """
    Responsibility: Loading, cleaning, tokenizing, and saving the corpus.
    """
    #Load all .txt files from a folder
    def load(self, folder_path):
        book_data = {}
        # check if the folder exists
        if not os.path.exists(folder_path):
            return book_data
        #if it exist then store in files list by files name 
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
        #start reading the books
        for filename in files:
            # the file called f please start to read it
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                book_data[filename] = f.read()
        return book_data


    #Remove Gutenberg header and footer
    def strip_gutenberg(self, text):
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        start_idx = text.find(start_marker)
        #check if the start marker exist
        if start_idx != -1:
            #find the idex after the start marker
            line_end = text.find("\n", start_idx)
            text = text[line_end:]
        end_idx = text.find(end_marker)
        if end_idx != -1:
            # last file index before the end 
            text = text[:end_idx]
        # text Vaccum cleaner by .strip   
        return text.strip()

    # lower case all text 
    def lowercase(self,text):
        return text.lower()
    
    # Remove all punctuation
    def remove_punctuation(self,text):
        text = re.sub(r'[^a-z\s]', ' ', text)
        return text
        
    # removing number including roman number
    def remove_numbers(self,text):
        text = re.sub(r'\b[ivxlcdm]+\b\.', '', text) 
        text = re.sub(r'\d+', '', text)
        return text

    # Remove extra whitespace and blank lines
    def remove_whitespace (self,text):
        text = " ".join(text.split())
        return text

    def normalize(self, text):
  
        # 1. lower case
        text=self.lowercase(text)
        #2remove number
        text=self.remove_numbers(text)
        #3 remove punctuation
        text=self.remove_punctuation(text)
        #4.remove whitespace
        text=self.remove_whitespace(text)

        return text

    3
    #Split text into a list of sentences
    # NTLK smart enough to know the end of the sentence not just a dot 
    def sentence_tokenize(self, text):
        return nltk.sent_tokenize(text)
    #Split a single sentence into a list of tokens
    def word_tokenize(self, sentence):
        return sentence.split()

    def save(self, tokenized_sentences, filepath):
        # 1. Create the folder if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # 2. Loop through every sentence list
        with open(filepath, 'w', encoding='utf-8') as f:
            # 3. Join the words back into a single string line
            for sentence_list in tokenized_sentences:
                line = " ".join(sentence_list)
                # 4. Write it to the file with a NEWLINE (\n)
                if line.strip():
                    f.write(line + "\n")

# --- USEFUL TEST BLOCK ---
if __name__ == "__main__":
    norm = Normalizer()
    
    # Test 1: Standard normalization
    print("Test 1 (Standard):", norm.normalize("Test 123: This is a sentence!"))
    
    # Test 2: Dash replacement (the observerexcellent fix)
    print("Test 2 (Dashes):", norm.normalize("observer—excellent"))
    print("Test 3 (Dashes):", norm.normalize("ix. i'm ahmed"))