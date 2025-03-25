from openai import OpenAI
from tqdm import tqdm
import json
import random
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

class LLMEvaluator:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
    
    def evaluate(self, prompt, response_a, response_b, method='baseline', item_id=None):
        """Enhanced evaluation with detailed logging"""
        if method == 'baseline':
            evaluation_prompt = f"""
            Prompt: {prompt}
            Response A: {response_a}
            Response B: {response_b}
            Which is better? Answer only 'A' or 'B':
            """
            system_msg = "You are comparing two responses."
            temp = 0.7
        elif method == 'cot':
            evaluation_prompt = f"""
            Let's think step by step:
            
            1. Understand the prompt: "{prompt}..."
            
            2. Evaluate Response A:
            {response_a}...
            - Accuracy: Any factual errors?
            - Completeness: Fully addresses prompt?
            - Helpfulness: Practical usefulness?
            
            3. Evaluate Response B:
            {response_b}...
            (Same criteria as above)
            
            4. Final comparison:
            Which response is better? Answer only 'A' or 'B':
            """
            system_msg = "Analyze responses step-by-step before choosing."
            temp = 0.7

        # # Print evaluation header
        # print(f"\n{Back.BLUE}EVALUATION {item_id or ''} ({method.upper()}){Style.RESET_ALL}")
        # print(f"{Fore.YELLOW}PROMPT:{Style.RESET_ALL}\n{prompt[:500]}{'...' if len(prompt)>500 else ''}")
        # print(f"\n{Fore.GREEN}CHOSEN RESPONSE:{Style.RESET_ALL}\n{response_a[:500]}{'...' if len(response_a)>500 else ''}")
        # print(f"\n{Fore.RED}REJECTED RESPONSE:{Style.RESET_ALL}\n{response_b[:500]}{'...' if len(response_b)>500 else ''}")

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=temp,
                max_tokens=10
            )
            judgement = response.choices[0].message.content.strip()
            
            # Print judgement with color
            judgement_color = Fore.GREEN if judgement == 'A' else Fore.RED
            print(f"\n{Fore.CYAN}JUDGEMENT:{Style.RESET_ALL} {judgement_color}{judgement}{Style.RESET_ALL}")
            
            return judgement
        except Exception as e:
            print(f"{Fore.RED}ERROR:{Style.RESET_ALL} {str(e)}")
            return None

def load_dataset(file_path, sample_size=100):
    """Load dataset with validation"""
    try:
        with open(file_path) as f:
            data = json.load(f)
        sample = random.sample(data, min(sample_size, len(data)))
        
        # Print dataset stats
        # print(f"\n{Fore.BLUE}DATASET INFO:{Style.RESET_ALL}")
        # print(f"Loaded {len(sample)}/{len(data)} items")
        # print(f"First prompt: {sample[0]['prompt'][:100]}...")
        return sample
    except Exception as e:
        print(f"{Fore.RED}LOAD ERROR:{Style.RESET_ALL} {str(e)}")
        return []

def run_evaluation(dataset, evaluator, method):
    """Run evaluation with full logging"""
    results = []
    details = []
    
    for idx, item in enumerate(tqdm(dataset, desc=f"Evaluating ({method})")):
        chosen = item["chosen"][0] if isinstance(item["chosen"], list) else item["chosen"]
        rejected = item["rejected"][0] if isinstance(item["rejected"], list) else item["rejected"]
        
        judgement = evaluator.evaluate(
            prompt=item["prompt"],
            response_a=chosen,
            response_b=rejected,
            method=method,
            item_id=idx+1
        )
        
        is_correct = judgement == 'A'
        results.append(is_correct)
        details.append({
            "id": idx+1,
            "prompt": item["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "judgement": judgement,
            "correct": is_correct
        })
        
        # Print immediate result
        result_color = Fore.GREEN if is_correct else Fore.RED
        print(f"{result_color}SAMPLE {idx+1} RESULT: {'CORRECT' if is_correct else 'INCORRECT'}{Style.RESET_ALL}")
        print(f"{'-'*50}")
    
    accuracy = sum(results) / len(results) if results else 0
    print(f"\n{Back.GREEN}FINAL {method.upper()} ACCURACY:{Style.RESET_ALL} {Fore.BLUE}{accuracy:.2%}{Style.RESET_ALL}")
    return accuracy, details

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "api_key": "sk-45e23ddf232e4b2bb626a17e64e221c2",
        "dataset_path": "total_dataset.json",
        "sample_size": 100,
        "methods": ["baseline", "cot"]
    }
    
    # Load data
    print(f"{Fore.CYAN}\nINITIALIZING EVALUATION...{Style.RESET_ALL}")
    dataset = load_dataset(CONFIG["dataset_path"], CONFIG["sample_size"])
    
    if not dataset:
        exit()

    # Initialize evaluator
    evaluator = LLMEvaluator(CONFIG["api_key"])
    
    # Run evaluations
    full_results = {}
    for method in CONFIG["methods"]:
        print(f"\n{Fore.MAGENTA}STARTING {method.upper()} EVALUATION{Style.RESET_ALL}")
        accuracy, details = run_evaluation(dataset, evaluator, method)
        full_results[method] = {
            "accuracy": accuracy,
            "details": details
        }
    
    # Save results
    with open("detailed_results.json", "w") as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n{Back.GREEN}EVALUATION COMPLETE{Style.RESET_ALL}")
    print(f"Detailed results saved to 'detailed_results.json'")
