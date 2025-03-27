import numpy as np
import pandas as pd
import json
import random
from datetime import datetime
from openai import OpenAI
from colorama import Fore, Style, init
from tqdm import tqdm
from collections import defaultdict

init(autoreset=True)

class GenRM:
    def __init__(self, api_key, base_url="https://api.deepseek.com/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "deepseek-chat"
    
    def load_dataset(self, file_path, sample_size=10):
        """Load dataset from JSON or Parquet with validation"""
        if file_path.endswith('.json'):
            with open(file_path) as f:
                data = json.load(f)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            data = df.to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use .json or .parquet")

        # Validate required columns
        sample = random.sample(data, min(sample_size, len(data)))
        for item in sample:
            if 'prompt' not in item or 'chosen' not in item or 'rejected' not in item:
                raise ValueError("Dataset missing required columns (prompt, chosen, rejected)")

        print(f"\n{Fore.GREEN}Loaded {len(sample)}/{len(data)} items{Style.RESET_ALL}")
        return sample

    def verify(self, prompt, solution, temp=0.7):
         """Get P(Yes) using softmax as per GenRM paper"""

         verification_prompt = f"""
         Is this answer correct? Answer strictly 'Yes' or 'No':
         
         Question: {prompt}
         Answer: {solution}
         """
         
         response = self.client.chat.completions.create(
               model=self.model,
               messages=[{"role": "user", "content": verification_prompt}],
               temperature=temp,
               max_tokens=1,
               logprobs=True,
               top_logprobs=5
         )
         
         # Softmax calculation
         top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
         probs = {lp.token.lower(): np.exp(lp.logprob) for lp in top_logprobs}
         p_yes = probs.get("yes", 0.0) / (probs.get("yes", 0.0) + probs.get("no", 0.0))
         return p_yes if not np.isnan(p_yes) else 0.0

        
    def verify_cot(self, prompt, solution, temp=0.7):
        cot_prompt = f"""
        [CoT Verification Protocol]
        Question: {prompt}
        Answer: {solution}

        Steps:
        1. Fact Check: Verify objective claims
        2. Logical Consistency: Check for contradictions
        3. Final Judgment: Conclude with EXACTLY '[[Yes]]' or '[[No]]'
        
        Analysis:"""
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": cot_prompt}],
            temperature=temp,
            max_tokens=150
        )
        output = response.choices[0].message.content
        verdict = '[[Yes]]' in output[-20:] 
        return {
            'verdict': verdict,
            'rationale': output  
        }
        

    def evaluate_dataset(self, dataset, sample_size=10, method="genrm", votes=3):
        """Unified evaluator with detailed CoT tracking"""
        results = []
        domain_errors = defaultdict(list)
        sample = random.sample(dataset, min(sample_size, len(dataset)))
        
        for idx, item in enumerate(tqdm(sample, desc=f"{method.upper()} Evaluation"), 1):
            prompt = str(item['prompt'])
            chosen = str(item['chosen'][0] if isinstance(item['chosen'], list) else item['chosen'])
            rejected = str(item['rejected'][0] if isinstance(item['rejected'], list) else item['rejected'])
            domain = str(item.get('domain', 'unknown'))

            # Verification
            chosen_p_yes = self.verify(prompt, chosen)
            rejected_p_yes = self.verify(prompt, rejected)
            result = {
                'type': 'direct',
                'p_yes_chosen': chosen_p_yes,
                'p_yes_rejected': rejected_p_yes,
                'chosen_correct': chosen_p_yes > 0.5,
                'rejected_correct': rejected_p_yes > 0.5,
            }

            # Result processing
            passed = result['chosen_correct'] and not result['rejected_correct']
            detail = {
                'id': idx,
                'prompt': prompt,
                'domain': domain,
                'method': method,
                'passed': passed,
                'score': result.get('avg_p_yes_chosen', result.get('p_yes_chosen', 0)),
                **result
            }
            results.append(detail)

            if not passed:
                domain_errors[domain].append({
                    'id': idx,
                    'reason': "Chosen rejected or incorrect accepted",
                    'chosen_verdict': result['chosen_correct'],
                    'rejected_verdict': result['rejected_correct']
                })

            # Print results
            color = Fore.GREEN if passed else Fore.RED
            status = "✓" if passed else "✗"
            if method == "genrm_cot":
                score = result['avg_p_yes_chosen']
            else:
                score = result['p_yes_chosen']
                
            print(f"{color}Item {idx}: {status} | Score: {score:.4f} | Method: {method} | Domain: {domain}")

        # Calculate accuracy
        accuracy = sum(r['passed'] for r in results) / len(results) if results else 0.0
        print(f"\n{Fore.BLUE}{method.upper()} RESULTS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Accuracy:{Style.RESET_ALL} {accuracy:.2%}")
        print(f"{Fore.YELLOW}Domain Errors:{Style.RESET_ALL} {sum(len(v) for v in domain_errors.values())}")

        return accuracy, results, dict(domain_errors)
      
    def save_results(self, results, domain_errors):
      """Convert NumPy types to native Python for JSON serialization"""
      def convert_types(obj):
         if isinstance(obj, (np.bool_, np.integer)):
               return bool(obj) if isinstance(obj, np.bool_) else int(obj)
         elif isinstance(obj, (np.floating, np.number)):
               return float(obj)
         elif isinstance(obj, np.ndarray):
               return obj.tolist()
         elif isinstance(obj, (list, tuple)):
               return [convert_types(item) for item in obj]
         elif isinstance(obj, dict):
               return {key: convert_types(value) for key, value in obj.items()}
         return obj

      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      output = {
         "metadata": {
               "evaluation_date": timestamp,
               "method": "genrm",
               "total_samples": len(results),
               "accuracy": float(sum(1 for r in results if r['passed'])) / len(results)
         },
         "results": convert_types(results),
         "domain_errors": convert_types(domain_errors)
      }
      
      filename = f"/root/dl/output/genrm_results_{timestamp}.json"
      with open(filename, 'w') as f:
         json.dump(output, f, indent=2)
      
      print(f"\n{Fore.GREEN}Results saved to:{Style.RESET_ALL} {filename}")
      return filename


if __name__ == "__main__":
    # Initialize with your API key
    genrm = GenRM(api_key="sk-63a7f9debd6c45ce8fc5ac25efffc162")
    
    # Load your dataset
    dataset = genrm.load_dataset(
        # file_path="dataset/train-00000-of-00001-2a1df75c6bce91ab.parquet",  
        file_path="dataset/total_dataset.json",  
        sample_size=1000
            # Load 100 items (we'll evaluate a subset)
    )
    
    # Evaluate on 10 random samples
    accuracy, results, domain_errors = genrm.evaluate_dataset(dataset, sample_size=5, votes=3)

    genrm.save_results(results, domain_errors)
    
    # Print sample results
   #  for i, r in enumerate(results[:3]):
      #   print(f"\n{i+1}. Prompt: {r['prompt']}")
      #   print(f"   ✓ Chosen: P(Yes)={r['p_yes_chosen']:.2f} | {r['chosen']}")
      #   print(f"   ✗ Rejected: P(Yes)={r['p_yes_rejected']:.2f} | {r['rejected']}")
