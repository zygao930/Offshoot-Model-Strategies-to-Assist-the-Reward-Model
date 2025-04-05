import pandas as pd
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from collections import defaultdict
import json
import string
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial
import os

class WordSortingDatasetGenerator:
    def __init__(self, model_path="/root/autodl-tmp/LLM-Research/gemma-7b"):
        """Initialize with proper multiprocessing setup"""
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        self.model_path = model_path
        self.tokenizer, self.model = self.init_model()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_model(self):
        """Initialize model with proper quantization config"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config
        ).eval()
        
        return tokenizer, model

    def load_names(self, csv_path="./dataset/word_sorting_words.csv"):
        """Load names from CSV"""
        return pd.read_csv(csv_path)["word"].tolist()

    @staticmethod
    def generate_sorted_answer(words):
        """Generate correct sorted answer"""
        return ' '.join(sorted(words, key=lambda x: x.lower()))

    def generate_problem_queries(self, names, lengths=[2, 3, 4], num_queries=4096):
        """Generate problem queries with deduplication"""
        queries = defaultdict(list)
        for length in lengths:
            seen = set()
            with tqdm(total=num_queries, desc=f"Generating {num_queries} queries of length {length}") as pbar:
                while len(queries[length]) < num_queries:
                    query = random.sample(names, length)
                    query_tuple = tuple(sorted(query))  # For hashability (order doesn't matter)
                    if query_tuple not in seen:
                        seen.add(query_tuple)
                        queries[length].append(query)
                        pbar.update(1)
        return queries

    def generate_attempts_batch(self, queries, num_attempts=16):
        """Generate sorting attempts for multiple queries at once (batch processing)"""
        prompt_variations = [
            "Given the words: {}. Sort them alphabetically:",
            "Words: {}. Alphabetical order:",
            "Arrange these words in alphabetical order: {}. Sorted:",
            "Sort the following words alphabetically: {}. Result:",
            "Please alphabetize these words: {}. Alphabetical order:"
        ]
        
        # Create prompts for all queries
        prompts = []
        for query in queries:
            prompt_template = random.choice(prompt_variations)
            prompts.append(prompt_template.format(', '.join(query)))
        
        # Duplicate each prompt for the number of attempts
        repeated_prompts = [p for p in prompts for _ in range(num_attempts)]
        
        inputs = self.tokenizer(
            repeated_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(self.model.device)
        
        # Dynamic generation parameters
        max_len = max(len(" ".join(q)) for q in queries)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_len + 10,
            temperature=random.uniform(1.5, 2.0),
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=random.uniform(1.5, 2.0),
            top_k=random.randint(30, 50),
            top_p=random.uniform(0.85, 0.98),
            no_repeat_ngram_size=2
        )
        
        # Process all attempts
        all_attempts = []
        prompt_lengths = [len(inputs['input_ids'][i]) for i in range(len(inputs['input_ids']))]
        
        for i, output in enumerate(outputs):
            generated = output[prompt_lengths[i]:]
            attempt = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            # Clean and normalize the attempt
            attempt = attempt.strip()
            attempt = attempt.split('\n')[0]  # Take only the first line
            attempt = attempt.split('.')[0]    # Take only before first period
            attempt = ' '.join([w.strip(string.punctuation) for w in attempt.split() if w.isalpha()])
            
            all_attempts.append(attempt.lower())
        
        # Group attempts by original query
        attempts_by_query = []
        for i in range(len(queries)):
            start = i * num_attempts
            end = start + num_attempts
            attempts_by_query.append(all_attempts[start:end])
        
        return attempts_by_query

    def process_batch(self, query_batch, num_attempts):
        """Process a batch of queries"""
        batch_results = []
        correct_answers = [self.generate_sorted_answer(q) for q in query_batch]
        attempts_lists = self.generate_attempts_batch(query_batch, num_attempts)
        
        for query, correct, attempts in zip(query_batch, correct_answers, attempts_lists):
            seen_attempts = {correct}
            unique_attempts = [correct]
            
            for attempt in attempts:
                if (attempt not in seen_attempts and all(w.isalpha() for w in attempt.split())):
                  #   len(attempt.split()) == len(query) and 
                  #   all(w.isalpha() for w in attempt.split())):
                    seen_attempts.add(attempt)
                    unique_attempts.append(attempt)
            
            for attempt in unique_attempts[:num_attempts]:
                batch_results.append({
                    "query": ' '.join(query),
                    "attempt": attempt,
                    "is_correct": attempt == correct,
                    "length": len(query),
                    "correct_answer": correct
                })
        
        return batch_results

    def generate_dataset(self, names, output_path="ws_train.jsonl", 
                        lengths=[2, 3, 4], num_queries=4096, num_attempts=64,
                        batch_size=16, num_workers=4):
        """Generate dataset with multiprocessing"""
        queries_dict = self.generate_problem_queries(names, lengths, num_queries)
        
        # Prepare batches for multiprocessing
        all_batches = []
        for length, query_list in queries_dict.items():
            for i in range(0, len(query_list), batch_size):
                all_batches.append((query_list[i:i+batch_size], num_attempts))
        
        # Process in parallel
        with Pool(num_workers, initializer=worker_init) as pool:
            results = list(tqdm(
                pool.imap(process_batch_wrapper, all_batches),
                total=len(all_batches),
                desc="Generating dataset"
            ))
        
        # Flatten results
        dataset = [item for sublist in results for item in sublist]
        df = pd.DataFrame(dataset)
        
        # Save to JSONL file
        with open(output_path, 'w') as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict()) + '\n')
        
        # Enhanced reporting
        print(f"\nSaved {len(df)} samples to {output_path}")
        print(f"Overall Accuracy: {df['is_correct'].mean():.2%}")
        print("\nLength-wise Statistics:")
        length_stats = df.groupby('length').agg(
            count=('length', 'size'),
            accuracy=('is_correct', 'mean')
        )
        print(length_stats)
        
        return df

# Worker initialization function for multiprocessing
def worker_init():
    """Initialize worker process"""
    # Set different CUDA devices for different workers if available
    worker_id = int(os.getpid() % torch.cuda.device_count())
    torch.cuda.set_device(worker_id)
    global worker_generator
    worker_generator = WordSortingDatasetGenerator()

def process_batch_wrapper(args):
    """Wrapper function for pool workers"""
    batch, num_attempts = args
    return worker_generator.process_batch(batch, num_attempts)

if __name__ == "__main__":
    generator = WordSortingDatasetGenerator(model_path="/root/autodl-tmp/LLM-Research/gemma-7b")
    names = generator.load_names("./dataset/word_sorting_words.csv")
    
    # Generate training data (lengths 2, 3, 4)
    train_df = generator.generate_dataset(
        names,
        "ws_train_l3.jsonl",
        lengths=[4],
        num_queries=4096,
        num_attempts=16,
        batch_size=16,
        num_workers=1  # Adjust based on your GPU memory
    )
