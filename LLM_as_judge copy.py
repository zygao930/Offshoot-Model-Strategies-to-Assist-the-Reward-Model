import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import json
import random
from colorama import Fore, Back, Style, init
from collections import defaultdict
import os

# Initialize colorama
init(autoreset=True)

class LLMEvaluator:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    
    def evaluate(self, prompt, response_a, response_b, method='baseline', temp=0.6, model="deepseek-chat", item_id=None):
        """Enhanced evaluation with detailed logging"""
        if method == 'baseline':
            evaluation_prompt = f"""
            Prompt: {prompt}
            Response A: {response_a}
            Response B: {response_b}
            Which is better? Answer only 'A' or 'B':
            """
            system_msg = "You are comparing two responses."

        elif method == 'cot':
            evaluation_prompt = f"""
            Let's think step by step:
            
            1. Understand the prompt: "{prompt}"
            
            2. Evaluate Response A:
            {response_a}
            - Accuracy: Any factual errors?
            - Completeness: Fully addresses prompt?
            - Helpfulness: Practical usefulness?
            
            3. Evaluate Response B:
            {response_b}
            (Same criteria as above)
            
            4. Final comparison:
            Which response is better? Answer only 'A' or 'B':
            """
            system_msg = """Analyze responses step-by-step, but your FINAL output must be exactly 'A' or 'B' with no additional text."""

        elif method == 'critic':
            evaluation_prompt = f"""
            As a critical evaluator, analyze from multiple perspectives:

            PROMPT: "{prompt}"
    
            ---
            RESPONSE A: {response_a}
            RESPONSE B: {response_b}
            ---
            
            Evaluation Rubric (Score each criterion 1-5):
            1. Factual Correctness: Verifiable accuracy
            2. Intent Match: Alignment with prompt
            3. Coherence: Logical flow and clarity
            4. Practical Value: Actionability/usefulness
            5. Safety: Absence of harmful content
            
            Final Decision:
            - Compare total scores
            - If tied, prioritize factual correctness
            - Output must be exactly 'A' or 'B'
            """
            system_msg = """You are an AI critic. Evaluate rigorously but output ONLY 'A' or 'B'."""

        elif method == 'sop':
            evaluation_prompt = f"""
            [SOP Evaluation Protocol]
            
            Prompt: "{prompt}"
            
            Responses:
            A: {response_a}
            B: {response_b}
            
            ---
            Evaluation Steps:
            1. Fact Check: Verify claims, disqualify if false
            2. Semantic Relevance: Check prompt coverage
            3. Style Consistency: Evaluate tone match
            
            Final Decision:
            - Output ONLY 'A' or 'B'
            """
            system_msg = """You are an SOP evaluation bot. Follow protocol but output ONLY 'A' or 'B'."""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=temp,
                max_tokens=10
            )
            judgement = response.choices[0].message.content.strip().upper()
            return judgement if judgement in ['A', 'B'] else None
        except Exception as e:
            print(f"{Fore.RED}ERROR:{Style.RESET_ALL} {str(e)}")
            return None

def load_dataset(file_path, sample_size=100):
    """Load dataset from JSON or Parquet with validation"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        if file_path.endswith('.json'):
            with open(file_path) as f:
                data = json.load(f)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            data = df.to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use .json or .parquet")

        if not data:
            raise ValueError("Dataset is empty")

        # Validate required columns
        sample = random.sample(data, min(sample_size, len(data)))
        for item in sample:
            if 'prompt' not in item or 'chosen' not in item or 'rejected' not in item:
                raise ValueError("Dataset missing required columns (prompt, chosen, rejected)")

        print(f"\n{Fore.GREEN}Loaded {len(sample)}/{len(data)} items{Style.RESET_ALL}")
        return sample

    except Exception as e:
        print(f"{Fore.RED}LOAD ERROR:{Style.RESET_ALL} {str(e)}")
        return []

def run_evaluation(dataset, evaluator, method):
    """Run evaluation with comprehensive tracking"""
    results = []
    details = []
    domain_errors = defaultdict(list)
    
    for idx, item in enumerate(tqdm(dataset, desc=f"{method.upper()} Evaluation")):
        try:
            prompt = str(item['prompt'])
            chosen = str(item['chosen'][0] if isinstance(item['chosen'], list) else item['chosen'])
            rejected = str(item['rejected'][0] if isinstance(item['rejected'], list) else item['rejected'])
            domain = str(item.get('domain', 'unknown'))

            judgement = evaluator.evaluate(
                prompt=prompt,
                response_a=chosen,
                response_b=rejected,
                method=method,
                item_id=idx+1
            )

            if judgement is None:
                continue

            is_correct = judgement == 'A'
            results.append(is_correct)
            
            detail = {
                'id': idx+1,
                'prompt': prompt[:500] + '...' if len(prompt) > 500 else prompt,
                'chosen': chosen[:500] + '...' if len(chosen) > 500 else chosen,
                'rejected': rejected[:500] + '...' if len(rejected) > 500 else rejected,
                'judgement': judgement,
                'correct': is_correct,
                'domain': domain
            }
            details.append(detail)

            if not is_correct:
                domain_errors[domain].append({
                    'id': idx+1,
                    'prompt': prompt,
                    'incorrect_judgement': judgement,
                    'chosen': chosen,
                    'rejected': rejected
                })

            # Print colored result
            color = Fore.GREEN if is_correct else Fore.RED
            print(f"{color}Item {idx+1}: {'✓' if is_correct else '✗'} {judgement}{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}Error processing item {idx+1}: {str(e)}{Style.RESET_ALL}")
            continue

    accuracy = sum(results)/len(results) if results else 0
    
    # Print summary
    print(f"\n{Back.BLUE} {method.upper()} RESULTS {Style.RESET_ALL}")
    print(f"{Fore.CYAN}Accuracy:{Style.RESET_ALL} {accuracy:.2%}")
    print(f"{Fore.YELLOW}Domain Errors:{Style.RESET_ALL}")
    for domain, errors in domain_errors.items():
        print(f"  {domain}: {len(errors)} errors")
    
    return accuracy, details, dict(domain_errors)

def save_results(results, domain_errors):
    """Save results with timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = f"evaluation_results_{timestamp}.json"
    errors_file = f"domain_errors_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(errors_file, 'w') as f:
        json.dump(domain_errors, f, indent=2)
    
    print(f"\n{Fore.GREEN}Results saved to:{Style.RESET_ALL}")
    print(f"- {results_file}")
    print(f"- {errors_file}")

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "api_key": ,
        "base_url": "https://api.deepseek.com/v1",
        "dataset_path": "dataset/train-00000-of-00001-2a1df75c6bce91ab.parquet",  # Update this path
        "sample_size": 100,
        "methods": ['baseline', 'cot', 'critic', 'sop'],
        "model": "deepseek-chat"
    }

    print(f"{Back.GREEN} LLM Response Evaluator {Style.RESET_ALL}\n")
    
    # Load data
    dataset = load_dataset(CONFIG["dataset_path"], CONFIG["sample_size"])
    if not dataset:
        exit(1)

    # Initialize evaluator
    evaluator = LLMEvaluator(CONFIG["api_key"], CONFIG["base_url"])
    
    # Run evaluations
    full_results = {}
    all_domain_errors = {}
    
    for method in CONFIG["methods"]:
        print(f"\n{Back.MAGENTA} {method.upper()} EVALUATION {'='*30}{Style.RESET_ALL}")
        accuracy, details, domain_errors = run_evaluation(dataset, evaluator, method)
        
        full_results[method] = {
            "accuracy": accuracy,
            "details": details,
            "domain_errors": domain_errors
        }
        all_domain_errors.update(domain_errors)
    
    # Save results
    save_results(full_results, all_domain_errors)
    
    print(f"\n{Back.GREEN} EVALUATION COMPLETE {Style.RESET_ALL}")
