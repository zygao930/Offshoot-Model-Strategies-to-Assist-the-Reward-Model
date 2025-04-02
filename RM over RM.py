import numpy as np
import pandas as pd
import json
import random
from datetime import datetime
from openai import OpenAI
from colorama import Fore, Style, init
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import Dataset
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model

init(autoreset=True)

class RMoverRM:
    def __init__(self, api_key, base_url="https://api.deepseek.com/v1", 
                 rm_model_path="/root/autodl-tmp/reward-model-deberta-v3-large-v2",  
                 device="cuda"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "deepseek-chat"
        self.rm_model_name = "deberta-v3-large"
        self.rm_model_path = rm_model_path  # Store local path
        self.device = device

        # Initialize RM model with explicit local_files_only
        self.rm_tokenizer = AutoTokenizer.from_pretrained(
            rm_model_path,
            local_files_only=True  # Critical for offline use
        )
        self.rm_model = AutoModelForSequenceClassification.from_pretrained(
            rm_model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True  # Critical for offline use
        ).to(device)
        
        self.cot_instructions = """
        Let's verify the answer step by step using Generative Evidence-based Chain-of-Thought (GEM-CoT):
         1. Fact Check: Verify claims, disqualify if false
         2. Semantic Relevance: Check prompt coverage
         3. Style Consistency: Evaluate tone match
         4. Practical Value: check Actionability/usefulness
         5. Safety: check if any harmful content
        
        Final Decision Rules:
        - MUST respond with exactly "[[✅]]" if ALL checks pass
        - MUST respond with exactly "[[❌]]" if ANY check fails
        - DO NOT include any other text, symbols, or explanations
        """
    
    def load_anthropic_dataset(self, train_path, test_path, sample_size=None):
        """Load Anthropic HH-RLHF parquet files with validation"""
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        # Convert to list of dicts (compatible with existing code)
        train_data = train_df.to_dict('records')
        test_data = test_df.to_dict('records')
        
        # Optional sampling
        if sample_size:
            train_data = random.sample(train_data, min(sample_size, len(train_data)))
            test_data = random.sample(test_data, min(sample_size//10, len(test_data)))  # Smaller test
        
        # Verify structure
        for item in train_data[:5] + test_data[:5]:
            if not all(k in item for k in ['chosen', 'rejected']):
                raise ValueError("Anthropic dataset missing required columns")
        
        print(f"{Fore.GREEN}Loaded Anthropic HH-RLHF: {len(train_data)} train, {len(test_data)} test{Style.RESET_ALL}")
        return train_data, test_data

    def generate_cot_dataset(self, dataset, sample_size=32):
        """Generate CoT verification dataset for RM training"""
        sample = random.sample(dataset, min(sample_size, len(dataset)))
        cot_data = []
        
        for item in tqdm(sample, desc="Generating CoT Dataset"):
            prompt = str(item['prompt'])
            chosen = str(item['chosen'][0] if isinstance(item['chosen'], list) else item['chosen'])
            rejected = str(item['rejected'][0] if isinstance(item['rejected'], list) else item['rejected'])
            
            # Generate CoT verification for chosen answer
            cot_prompt = f"""
            [CoT Verification Protocol]
            Question: {prompt}
            Proposed Answer: {chosen}
            {self.cot_instructions}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": cot_prompt}],
                temperature=0.3,
                max_tokens=256
            )
            
            cot_verdict = response.choices[0].message.content.strip()
            is_correct = 1 if "[[✅]]" in cot_verdict else 0
            
            cot_data.append({
                'prompt': prompt,
                'response': chosen,
                'cot_verdict': cot_verdict,
                'label': is_correct
            })
            
            # Generate CoT verification for rejected answer
            cot_prompt = f"""
            [CoT Verification Protocol]
            Question: {prompt}
            Proposed Answer: {rejected}
            {self.cot_instructions}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": cot_prompt}],
                temperature=0.3,
                max_tokens=256
            )
            
            cot_verdict = response.choices[0].message.content.strip()
            is_correct = 1 if "[[✅]]" in cot_verdict else 0
            
            cot_data.append({
                'prompt': prompt,
                'response': rejected,
                'cot_verdict': cot_verdict,
                'label': is_correct
            })
        
        return cot_data

    def train_rm_model(self, cot_data, epochs=4, lr=1e-5):
      """Train RM model using KTO (Kahneman-Tversky Optimization)"""
      # 1. Prepare dataset with proper label conversion
      df = pd.DataFrame(cot_data)
      dataset = Dataset.from_pandas(df)

      # 2. Tokenization 
      def tokenize_function(examples):
         prompts = [p[0] if isinstance(p, list) else str(p) for p in examples['prompt']]
         responses = [r[0] if isinstance(r, list) else str(r) for r in examples['response']]
         
         # Combine with separator
         texts = [
            prompt + self.rm_tokenizer.sep_token + response 
            for prompt, response in zip(prompts, responses)
         ]
         
         # Tokenize and convert labels to float32
         tokenized = self.rm_tokenizer(
               texts,
               padding="max_length",
               truncation=True,
               max_length=512,
               return_tensors="pt"
         )
         
         # Convert labels to float32
         tokenized['labels'] = torch.tensor(examples['label'], dtype=torch.float32)
         return tokenized
      
      tokenized_dataset = dataset.map(tokenize_function, batched=True)
      tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
      
      # Configure LoRA for efficient training
      lora_config = LoraConfig(
         r=16,
         lora_alpha=32,
         target_modules=["query_proj", "value_proj"],
         lora_dropout=0.05,
         bias="none",
         task_type="SEQ_CLS"
      )
      
      model = get_peft_model(self.rm_model, lora_config)
      model.print_trainable_parameters()
      
      training_args = TrainingArguments(
         output_dir="./rm_checkpoints",
         evaluation_strategy="epoch",
         learning_rate=lr,
         per_device_train_batch_size=8,
         per_device_eval_batch_size=8,
         num_train_epochs=epochs,
         weight_decay=0.01,
         save_strategy="epoch",
         load_best_model_at_end=True,
         metric_for_best_model="accuracy",
         fp16=False,  
         bf16=True
      )
      
      def compute_metrics(eval_pred):
         logits, labels = eval_pred
         predictions = np.argmax(logits, axis=-1)
         return {"accuracy": (predictions == labels).mean()}
      
      trainer = Trainer(
         model=model,
         args=training_args,
         train_dataset=tokenized_dataset["train"],
         eval_dataset=tokenized_dataset["test"],
         compute_metrics=compute_metrics
      )
      
      trainer.train()
      return model

    def rm_scoring(self, prompt, response):
      """Score response using trained RM model"""
      inputs = self.rm_tokenizer(
         prompt + self.rm_tokenizer.sep_token + response,
         return_tensors="pt",
         padding=True,
         truncation=True,
         max_length=512
      ).to(self.device)
      
      with torch.no_grad():
         outputs = self.rm_model(**inputs)
         logits = outputs.logits
         # Convert to float32 before sigmoid if using BFloat16
         if logits.dtype == torch.bfloat16:
            logits = logits.float()
         scores = torch.sigmoid(logits).cpu().numpy()
      
      return float(scores[0][0])  # Return single score

    def evaluate_with_rm(self, dataset, sample_size=10):
        """Evaluate dataset using trained RM instead of majority vote"""
        results = []
        sample = random.sample(dataset, min(sample_size, len(dataset)))
        
        for idx, item in enumerate(tqdm(sample, desc="RM Evaluation"), 1):
            prompt = str(item['prompt'])
            chosen = str(item['chosen'][0] if isinstance(item['chosen'], list) else item['chosen'])
            rejected = str(item['rejected'][0] if isinstance(item['rejected'], list) else item['rejected'])
            domain = str(item.get('domain', 'unknown'))
            
            # Score responses using RM
            chosen_score = self.rm_scoring(prompt, chosen)
            rejected_score = self.rm_scoring(prompt, rejected)
            
            # Determine if the pair is correct (chosen should score higher)
            passed = chosen_score > rejected_score
            
            results.append({
                'id': idx,
                'prompt': prompt,
               #  'domain': domain,
                'passed': passed,
                'chosen_score': chosen_score,
                'rejected_score': rejected_score,
                'score_diff': chosen_score - rejected_score
            })
            
            color = Fore.GREEN if passed else Fore.RED
            print(f"{color}Item {idx}: {'✓' if passed else '✗'} | "
                  f"Chosen: {chosen_score:.4f} | Rejected: {rejected_score:.4f} | "
                  f"Diff: {chosen_score - rejected_score:.4f} | Domain: {domain}")
        
        accuracy = sum(r['passed'] for r in results) / len(results) if results else 0.0
        print(f"\n{Fore.BLUE}RM-over-RM RESULTS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Accuracy:{Style.RESET_ALL} {accuracy:.2%}")
        print(f"{Fore.YELLOW}Avg Chosen Score:{Style.RESET_ALL} {np.mean([r['chosen_score'] for r in results]):.4f}")
        print(f"{Fore.YELLOW}Avg Rejected Score:{Style.RESET_ALL} {np.mean([r['rejected_score'] for r in results]):.4f}")
        return accuracy, results

    def save_results(self, results, method="rm"):
        """Save evaluation results with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "metadata": {
                "evaluation_date": timestamp,
                "model": self.rm_model_name if method == "rm" else "deepseek-chat",
                "method": method,
                "total_samples": len(results),
                "passed_samples": sum(1 for r in results if r['passed']),
                "accuracy": sum(1 for r in results if r['passed']) / len(results),
                "avg_chosen_score": np.mean([r['chosen_score'] for r in results]),
                "avg_rejected_score": np.mean([r['rejected_score'] for r in results])
            },
            "results": results
        }
        
        filename = f"output/rm_over_rm_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n{Fore.GREEN}Results saved to:{Style.RESET_ALL} {filename}")
        return filename

if __name__ == "__main__":
    # Initialize with your API key
    rm_system = RMoverRM(api_key="sk-63a7f9debd6c45ce8fc5ac25efffc162")
    
    # 1. Load dataset
    train_data, test_data = rm_system.load_anthropic_dataset(
        train_path="./dataset/anthropic-hh-rlhf/train-00000-of-00001.parquet",
        test_path="./dataset/anthropic-hh-rlhf/test-00000-of-00001.parquet",
        sample_size=1000  # Optional: reduce for testing
    )
    
    # 2. Generate CoT dataset (32 samples)
    cot_data = rm_system.generate_cot_dataset(train_data, sample_size=100)
    
    # 3. Train RM model on CoT data
    trained_rm = rm_system.train_rm_model(cot_data)
    
    # 4. Evaluate with trained RM
    accuracy, results = rm_system.evaluate_with_rm(test_data, sample_size=100)
    
    # 5. Save results
    rm_system.save_results(results)
