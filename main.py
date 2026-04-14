import os
import argparse
import logging
from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

def main():
    load_dotenv(dotenv_path="config/.env")
    
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    # Argument Parsing
    parser = argparse.ArgumentParser(description="Sherlock N-Gram Predictor")
    parser.add_argument("--step", type=str, required=True, 
                        choices=["dataprep", "model", "inference", "all"])
    args = parser.parse_args()

    # Initialize Shared Instances
    norm = Normalizer()
    model = NGramModel()

    # --- 1. DATA PREPARATION ---
    if args.step in ["dataprep", "all"]:
        logger.info("Step 1: Starting Data Preparation...")
        raw_dir = os.getenv("TRAIN_RAW_DIR")
        output_file = os.getenv("TRAIN_TOKENS")
        
        raw_books = norm.load(raw_dir)
        all_sentences = []
        for filename, text in raw_books.items():
            clean_text = norm.strip_gutenberg(text)
            sentences = norm.sentence_tokenize(clean_text)
            for s in sentences:
                normalized = norm.normalize(s)
                tokens = norm.word_tokenize(normalized)
                if tokens: all_sentences.append(tokens)
        
        norm.save(all_sentences, output_file)
        logger.info(f"Data Prep complete. Tokens saved to {output_file}")

    # --- 2. MODEL TRAINING ---
    if args.step in ["model", "all"]:
        logger.info("Step 2: Starting Model Training...")
        token_file = os.getenv("TRAIN_TOKENS")
        
        # Ensure directory exists for saving
        os.makedirs(os.path.dirname(os.getenv("MODEL")), exist_ok=True)
        
        # Build and Save
        model.build_vocab(token_file, threshold=int(os.getenv("UNK_THRESHOLD", 3)))
        model.build_counts_and_probabilities(token_file)
        model.save_model(os.getenv("MODEL"))
        model.save_vocab(os.getenv("VOCAB"))
        logger.info("Model Training complete and files saved.")

    # --- 3. INFERENCE ---
    if args.step in ["inference", "all"]:
        logger.info("Step 3: Entering Inference Mode...")
        
        # Load the newly created files
        model.load(os.getenv("MODEL"), os.getenv("VOCAB"))
        predictor = Predictor(model, norm)
        
        print("\n" + "="*40)
        print("   SHERLOCK HOLMES NEXT-WORD PREDICTOR")
        print("="*40)
        print("Enter a phrase to see predictions (or 'q' to quit).")

        while True:
            text = input("\nContext: ").strip()
            if text.lower() in ['q', 'quit', 'exit']:
                break
            if not text:
                continue
                
            suggestions = predictor.predict_next(text, k=int(os.getenv("TOP_K", 3)))
            print(f"Top {os.getenv('TOP_K')} Predictions: {suggestions}")

if __name__ == "__main__":
    main()