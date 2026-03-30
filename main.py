import os
import argparse
import logging
from dotenv import load_dotenv

# Import custom modules
from src.data_prep.normalizer import Normalizer

def main():
    # 1. Setup and Config
    load_dotenv(dotenv_path="config/.env")
    
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    # 2. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="N-Gram Next-Word Predictor")
    parser.add_argument("--step", type=str, required=True, 
                        choices=["dataprep", "model", "inference", "all"])
    args = parser.parse_args()

    # 3. Initialize Normalizer
    norm = Normalizer()

    # --- STEP: DATA PREPARATION ---
    if args.step == "dataprep" or args.step == "all":
        logger.info("Starting Data Preparation...")
        raw_dir = os.getenv("TRAIN_RAW_DIR")
        output_file = os.getenv("TRAIN_TOKENS")

        if not raw_dir or not output_file:
            logger.error("Config variables missing. Check config/.env")
            return

        # Load returns a dictionary: { "Book1.txt": "full text...", ... }
        raw_books_dict = norm.load(raw_dir)
        all_tokenized_sentences = []

        # Iterate through each book individually
        for filename, raw_text in raw_books_dict.items():
            logger.info(f"Processing {filename}...")
            
            # Clean THIS book specifically
            clean_book_text = norm.strip_gutenberg(raw_text)
            
            # Split THIS book into sentences
            sentences = norm.sentence_tokenize(clean_book_text)
            
            for s in sentences:
                # Normalize (lowercase, remove punc/nums/dashes)
                cleaned_s = norm.normalize(s)
                tokens = norm.word_tokenize(cleaned_s)
                
                if tokens:
                    all_tokenized_sentences.append(tokens)

        # Save: This method handles the "\n" for one sentence per line
        norm.save(all_tokenized_sentences, output_file)
        logger.info(f"Success! Saved {len(all_tokenized_sentences)} sentences to {output_file}")

    # --- STEP: MODEL TRAINING ---
    if args.step == "model" or args.step == "all":
        logger.info("Model Training step is next!")

if __name__ == "__main__":
    main()