import pandas as pd
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import defaultdict

class LLCDatasetGenerator:
    def __init__(self, model_path="/root/autodl-tmp/LLM-Research/gemma-7b"):
        """Initialize with direct prompt formatting (no chat template)"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16  # Add this to match compute dtype
        ).eval()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_names(self, csv_path="combined_shuffled_names.csv"):
        """Load names from CSV"""
        return pd.read_csv(csv_path)["Name"].tolist()

    def generate_last_letter_concatenation(self, words):
        """Generate correct answer"""
        return "".join([word[-1] for word in words])

    def generate_problem_queries(self, names, lengths=[2], num_queries=10):
        """Generate problem queries with deduplication"""
        queries = defaultdict(list)
        for length in lengths:
            seen = set()
            while len(queries[length]) < num_queries:
                query = random.sample(names, length)
                query_tuple = tuple(query)  # For hashability
                if query_tuple not in seen:
                    seen.add(query_tuple)
                    queries[length].append(query)
        return queries

    def generate_attempts_batched(self, query, num_attempts=10):
        """Generate attempts with more dynamic outputs"""
        prompt_variations = [
            f"Given: {' '.join(query)}. Last letters concatenated:",
            f"Names: {' '.join(query)}. Combine last letters:",
            f"Take last letters from {' '.join(query)} and join them:",
            f"Final letters of {' '.join(query)} combined:"
        ]
        
        prompt = random.choice(prompt_variations)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        ).to(self.model.device)
        
        # Dynamic generation parameters
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=len(query) + 2,  # Slightly more than needed
            temperature=random.uniform(1.0, 1.5),  # Variable temperature
            do_sample=True,
            num_return_sequences=num_attempts,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=random.uniform(1.5, 2.0),
            top_k=random.randint(30, 50),
            top_p=random.uniform(0.85, 0.98),
            no_repeat_ngram_size=2
        )
        
        # Process outputs
        attempts = []
        prompt_length = len(inputs['input_ids'][0])
        for output in outputs:
            generated = output[prompt_length:]
            attempt = self.tokenizer.decode(generated, skip_special_tokens=True)
            attempt = ''.join([c.lower() for c in attempt if c.isalpha()])  # Clean without length restriction
            attempts.append(attempt)
        
        return attempts

    def generate_dataset(self, names, output_path="llc_train.jsonl", lengths=[2], num_queries=10, num_attempts=10):
        """Generate dataset with improved validation"""
        queries_dict = self.generate_problem_queries(names, lengths, num_queries)
        dataset = []
        
        for length, query_list in queries_dict.items():
            for query in tqdm(query_list, desc=f"Length {length}"):
                correct_answer = self.generate_last_letter_concatenation(query).lower()
                attempts = []
                
                # First add the correct answer exactly once
                attempts.append(correct_answer)
                
                # Then generate remaining attempts
                remaining_attempts = num_attempts - 1
                batch_size = 16
                for _ in range(0, remaining_attempts, batch_size):
                    current_batch = min(batch_size, remaining_attempts - len(attempts) + 1)
                    batch_attempts = self.generate_attempts_batched(query, current_batch)
                    attempts.extend(batch_attempts)
                
                # Deduplicate while preserving order and ensuring correct answer stays first
                seen_attempts = {correct_answer}
                unique_attempts = [correct_answer]  # Keep correct answer as first attempt
                
                for attempt in attempts[1:]:  # Skip first since we already added it
                    if attempt not in seen_attempts:
                        seen_attempts.add(attempt)
                        unique_attempts.append(attempt)
                
                # Store results (only up to num_attempts)
                for attempt in unique_attempts[:num_attempts]:
                    dataset.append({
                        "query": " ".join(query),
                        "attempt": attempt,
                        "is_correct": attempt == correct_answer,
                        "length": length,
                        "correct_answer": correct_answer,
                        "attempt_length": len(attempt)
                    })
        
        # Save with validation
        df = pd.DataFrame(dataset)
        # Filter bad attempts (must be alphabetic and within reasonable length)
        df = df[df['attempt'].apply(lambda x: x.isalpha() and 1 <= len(x) <= 10)]
        df = df.drop_duplicates(subset=["query", "attempt"])
        
        # Verify we have exactly one correct answer per query
        correct_counts = df[df['is_correct']].groupby('query').size()
        if any(correct_counts != 1):
            print("Warning: Some queries don't have exactly one correct answer")
        
        df.to_json(output_path, orient="records", lines=True)
        
        # Enhanced reporting
        print(f"Saved {len(df)} samples to {output_path}")
        print(f"Overall Accuracy: {df['is_correct'].mean():.2%}")
        print("\nLength-wise Accuracy:")
        print(df.groupby('length')['is_correct'].mean())
        print("\nAttempt Length Distribution:")
        print(df['attempt_length'].value_counts().sort_index())

if __name__ == "__main__":
    generator = LLCDatasetGenerator(model_path="/root/autodl-tmp/LLM-Research/gemma-7b")
    names = generator.load_names("./dataset/combined_shuffled_names.csv")
    
    generator.generate_dataset(
        names,
        "llc_train.jsonl",
        lengths=[2, 3, 4],  # Multiple lengths
        num_queries=350,
        num_attempts=128
    )
