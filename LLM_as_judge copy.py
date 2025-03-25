from openai import OpenAI
from tqdm import tqdm
import json
import random
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

class LLMEvaluator:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    
    def evaluate(self, prompt, response_a, response_b, method='baseline', temp=0.7, model="deepseek-chat", item_id=None):
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
            system_msg = """"Analyze responses step-by-step, but your FINAL output must be exactly 'A' or 'B' with no additional text. Never repeat instructions."""

        elif method == 'critic':
            evaluation_prompt = f"""
            As a critical evaluator, analyze from multiple perspectives:

            PROMPT: "{prompt}"
    
            ---
            RESPONSE A: {response_a}
            RESPONSE B: {response_b}
            ---
            
            Evaluation Rubric (Score each criterion 1-5):
            1. 𝗙𝗮𝗰𝘁𝘂𝗮𝗹 𝗖𝗼𝗿𝗿𝗲𝗰𝘁𝗻𝗲𝘀𝘀: Verifiable accuracy of information
            2. 𝗜𝗻𝘁𝗲𝗻𝘁 𝗠𝗮𝘁𝗰𝗵: Alignment with prompt requirements
            3. 𝗖𝗼𝗵𝗲𝗿𝗲𝗻𝗰𝗲: Logical flow and clarity
            4. 𝗣𝗿𝗮𝗰𝘁𝗶𝗰𝗮𝗹 𝗩𝗮𝗹𝘂𝗲: Actionability/usefulness
            5. 𝗦𝗮𝗳𝗲𝘁𝘆: Absence of harmful content
            
            Final Decision:
            - Compare total scores
            - If tied, prioritize factual correctness
            - Output must be exactly 'A' or 'B'
            """
            
            system_msg = """You are an AI critic. Rigorously evaluate responses using the rubric. 
            After scoring, output ONLY the better response's letter ('A' or 'B') with:
            - No explanations
            - No additional text
            - No score breakdowns"""

        elif method == 'sop':
            evaluation_prompt = f"""
            [SOP Evaluation Protocol]
            
            Prompt: "{prompt}"
            
            Responses:
            A: {response_a}
            B: {response_b}
            
            ---
            Follow these exact evaluation steps:
            
            1. 𝗙𝗮𝗰𝘁 𝗖𝗵𝗲𝗰𝗸 (Primary Priority):
            - Verify all claims against ground truth
            - Deduct points for any factual inaccuracies
            - If either response contains false information, automatically disqualify it
            
            2. 𝗦𝗲𝗺𝗮𝗻𝘁𝗶𝗰 𝗥𝗲𝗹𝗲𝘃𝗮𝗻𝗰𝗲:
            - Check if ALL aspects of the prompt are addressed
            - Score based on:
            • Coverage of required points (0-5)
            • Avoidance of irrelevant content
            
            3. 𝗦𝘁𝘆𝗹𝗲 𝗖𝗼𝗻𝘀𝗶𝘀𝘁𝗲𝗻𝗰𝘆:
            - Evaluate tone/formality match with prompt
            - Check for:
            • Appropriate terminology
            • Consistent persona/voice
            • Professional formatting (if applicable)
            
            Final Decision:
            - Fact errors → immediate rejection
            - Tiebreaker order: Relevance > Style
            - Output ONLY 'A' or 'B'
            """
            
            system_msg = """You are an SOP evaluation bot. RULES:
            1. INTERNALLY follow all evaluation steps
            2. EXTERNALLY output ONLY 'A' or 'B'
            3. NEVER explain your reasoning
            4. NEVER output evaluation steps"""

        # Very low temperature for strict adherence# # Print evaluation header
        # print(f"\n{Back.BLUE}EVALUATION {item_id or ''} ({method.upper()}){Style.RESET_ALL}")
        # print(f"{Fore.YELLOW}PROMPT:{Style.RESET_ALL}\n{prompt[:500]}{'...' if len(prompt)>500 else ''}")
        # print(f"\n{Fore.GREEN}CHOSEN RESPONSE:{Style.RESET_ALL}\n{response_a[:500]}{'...' if len(response_a)>500 else ''}")
        # print(f"\n{Fore.RED}REJECTED RESPONSE:{Style.RESET_ALL}\n{response_b[:500]}{'...' if len(response_b)>500 else ''}")

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
    """Run evaluation with full logging and domain-specific error tracking"""
    results = []
    details = []
    domain_errors = {}  # Track incorrect judgments by domain
    
    for idx, item in enumerate(tqdm(dataset, desc=f"Evaluating ({method})")):
        chosen = item["chosen"][0] if isinstance(item["chosen"], list) else item["chosen"]
        rejected = item["rejected"][0] if isinstance(item["rejected"], list) else item["rejected"]
        domain = item.get("domain", "unknown")  # Get domain or default to "unknown"
        
        judgement = evaluator.evaluate(
            prompt=item["prompt"],
            response_a=chosen,
            response_b=rejected,
            method=method,
            item_id=idx+1
        )
        
        is_correct = judgement == 'A'
        results.append(is_correct)
        
        # Update domain error tracking
        if not is_correct:
            if domain not in domain_errors:
                domain_errors[domain] = []
            domain_errors[domain].append({
                "id": item.get("id", idx+1),
                "prompt": item["prompt"],
                "incorrect_judgement": judgement,
                "chosen": chosen,
                "rejected": rejected
            })
        
        details.append({
            "id": idx+1,
            "prompt": item["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "judgement": judgement,
            "correct": is_correct,
            "domain": domain
        })
        
        # Print immediate result
        result_color = Fore.GREEN if is_correct else Fore.RED
        print(f"{result_color}SAMPLE {idx+1} RESULT: {'CORRECT' if is_correct else 'INCORRECT'}{Style.RESET_ALL}")
        print(f"{'-'*50}")
    
    accuracy = sum(results) / len(results) if results else 0
    
    # Print domain-specific error statistics
    print(f"\n{Back.RED}DOMAIN ERROR ANALYSIS:{Style.RESET_ALL}")
    for domain, errors in domain_errors.items():
        print(f"{Fore.YELLOW}{domain.upper()}:{Style.RESET_ALL} {len(errors)} errors")
    
    print(f"\n{Back.GREEN}FINAL {method.upper()} ACCURACY:{Style.RESET_ALL} {Fore.BLUE}{accuracy:.2%}{Style.RESET_ALL}")
    return accuracy, details, domain_errors  

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "api_key": 
        "base_url": "https://api.deepseek.com/v1",
        "dataset_path": "total_dataset.json",
        "sample_size": 100,
        "methods": ['baseline','cot','critic','sop']
    }
    
    # Load data
    print(f"{Fore.CYAN}\nINITIALIZING EVALUATION...{Style.RESET_ALL}")
    dataset = load_dataset(CONFIG["dataset_path"], CONFIG["sample_size"])

    # Initialize evaluator
    evaluator = LLMEvaluator(CONFIG["api_key"], CONFIG["base_url"])
    
    # Run evaluations
    full_results = {}
    domain_errors = {}  # Aggregate across all methods if needed
    
    for method in CONFIG["methods"]:
        print(f"\n{Fore.MAGENTA}STARTING {method.upper()} EVALUATION{Style.RESET_ALL}")
        accuracy, details, method_domain_errors = run_evaluation(dataset, evaluator, method)
        full_results[method] = {
            "accuracy": accuracy,
            "details": details,
            "domain_errors": method_domain_errors
        }
        domain_errors.update(method_domain_errors)
    
    # Save results with domain error details
    with open("detailed_results.json", "w") as f:
        json.dump(full_results, f, indent=2)
    
    # Optional: Save just the error analysis
    with open("domain_errors.json", "w") as f:
        json.dump(domain_errors, f, indent=2)
    
    print(f"\n{Back.GREEN}EVALUATION COMPLETE{Style.RESET_ALL}")
