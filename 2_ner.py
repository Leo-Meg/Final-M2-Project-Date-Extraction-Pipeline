from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import sys
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ner_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def print_gpu_utilization():
    if torch.cuda.is_available():
        logging.info(f"GPU utilization: {torch.cuda.utilization()}%")
        logging.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

class TextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe
        logging.info(f"Dataset initialized with {len(dataframe)} rows")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        """return (file_name, raw_text_content, original_url)"""
        row = self.data.iloc[idx]
        return (
            row['local_filename'],
            row['raw_text_content'],  # raw_text_content
            row['text version']
        )

def custom_collate_fn(batch: List[Tuple[str, str, str]]) -> Tuple[List[str], List[str], List[str]]:
    """Organizing batch data"""
    file_names, texts, urls = zip(*batch)
    return list(file_names), list(texts), list(urls)

class OptimizedNERProcessor:
    def __init__(
            self,
            model_name: str = "Jean-Baptiste/camembert-ner-with-dates",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            batch_size: int = 64,
            num_workers: int = 8
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = 512  # tokenizer maximum length

        logging.info(f"Initializing NER processor with device: {device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

            self.nlp = pipeline(
                task='ner',
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                aggregation_strategy='simple'
            )
            logging.info("Model loaded successfully")
            print_gpu_utilization()

        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise

    def process_text(self, text: str) -> List[str]:
        """Process a single text and return a list of dates"""
        try:
            # Process long text in chunks
            dates = []
            chunks = [text[i:i + self.max_length] for i in range(0, len(text), self.max_length)]

            for chunk in chunks:
                with torch.no_grad():
                    results = self.nlp(chunk)

                    chunk_dates = [
                        item['word']
                        for item in results
                        if isinstance(item, dict)
                           and item.get('entity_group') == 'DATE'
                    ]
                    dates.extend(chunk_dates)

            return self.filter_dates(dates)

        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            return []

    def filter_dates(self, dates: List[str]) -> List[str]:
        """Filtering and cleaning dates"""
        filtered_dates = [
            date for date in dates
            if 5 < len(date) < 20
               and "à" not in date
               and any(char.isdigit() for char in date)
        ]
        # Remove duplicates and keep order
        return list(dict.fromkeys(filtered_dates))[:15]

    def process_batch(self, file_names: List[str], texts: List[str], urls: List[str]) -> Dict[str, List[str]]:
        """Process a batch of text, returning a mapping from filenames to lists of dates"""
        results = {}
        for fname, text, url in zip(file_names, texts, urls):
            try:
                dates = self.process_text(text)
                logging.info(f"File: {fname}, Found {len(dates)} dates")
                results[fname] = dates
            except Exception as e:
                logging.error(f"Error processing file {fname}: {e}")
                results[fname] = []
        return results

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processing the entire DataFrame"""
        try:
            dataset = TextDataset(df[['local_filename', 'raw_text_content', 'text version']])
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=custom_collate_fn,
                pin_memory=True
            )

            all_results = {}
            for file_names, raw_texts, urls in tqdm(dataloader, desc="Processing files"):
                batch_results = self.process_batch(file_names, raw_texts, urls)
                all_results.update(batch_results)

                if len(all_results) % 50 == 0:
                    print_gpu_utilization()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

            df['time_list'] = df['local_filename'].map(all_results)
            return df

        except Exception as e:
            logging.error(f"Error in process_dataframe: {e}")
            raise

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path")
    parser.add_argument("--model", type=str, default="Jean-Baptiste/camembert-ner-with-dates")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    try:
        # Reading Data
        df = pd.read_csv(args.csv)
        logging.info(f"Loaded DataFrame with {len(df)} rows")

        # Initialize the processor
        processor = OptimizedNERProcessor(
            model_name=args.model,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Processing Data
        result_df = processor.process_dataframe(df)

        # Save the results
        output_path = Path(args.csv).stem + "_ner.csv"
        result_df.to_csv(output_path, index=False)
        logging.info(f"Results saved to {output_path}")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
