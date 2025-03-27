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

init(autoreset=True)

class GenRM:
    def __init__(self, api_key, base_url="https://api.deepseek.com/v1", confidence_threshold=0.5):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "deepseek-chat"
        self.confidence_threshold = confidence_threshold
        self.verification_instructions = """
        Let's verify step by step:
            1. Fact Check: Verify claims, disqualify if false
            2. Semantic Relevance: Check prompt coverage
            3. Style Consistency: Evaluate tone match
            
            Final Decision:
            - Output ONLY [[Yes]] or [[No]]
         """
    
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

        sample = random.sample(data, min(sample_size, len(data)))
        for item in sample:
            if 'prompt' not in item or 'chosen' not in item or 'rejected' not in item:
                raise ValueError("Dataset missing required columns (prompt, chosen, rejected)")

        print(f"\n{Fore.GREEN}Loaded {len(sample)}/{len(data)} items{Style.RESET_ALL}")
        return sample

    def _single_verification(self, prompt, solution):
      """Single CoT verification with proper probability extraction"""
      verification_prompt = f"""
      [CoT Verification Protocol]
      Question: {prompt}
      Proposed Answer: {solution}

      {self.verification_instructions}
      """
      
      response = self.client.chat.completions.create(
         model=self.model,
         messages=[{"role": "user", "content": verification_prompt}],
         temperature=0.7,
         max_tokens=300,
         logprobs=True,
         top_logprobs=5
      )
      
      # Debug: Print raw logprobs
      # print("\nRaw Logprobs Structure:")
      # for i, token in enumerate(response.choices[0].logprobs.content):
      #    print(f"Token {i}: {token.token} (logprob={token.logprob})")
      #    for top in token.top_logprobs:
      #          print(f"  Top: {top.token} (logprob={top.logprob})")

      # Extract probabilities for [[Yes]] and [[No]]
      yes_logprobs = []
      no_logprobs = []
      
      for token in response.choices[0].logprobs.content:
         for top in token.top_logprobs:
               if 'yes' in top.token.lower():
                  yes_logprobs.append(top.logprob)
               elif 'no' in top.token.lower():
                  no_logprobs.append(top.logprob)
      
      # Convert logprobs to probabilities
      if yes_logprobs and no_logprobs:
         max_yes = max(yes_logprobs)
         max_no = max(no_logprobs)
         p_yes = np.exp(max_yes) / (np.exp(max_yes) + np.exp(max_no))
      else:
         p_yes = 0.0  # Default to rejection if no verdict found

      print(f"\nCalculated P(Yes): {p_yes:.2%}")
      return p_yes, response.choices[0].message.content

    def verify_cot_majority(self, prompt, solution, votes):
        """Paper-accurate majority voting implementation"""
        with ThreadPoolExecutor(max_workers=min(votes, 5)) as executor:
            futures = [executor.submit(self._single_verification, prompt, solution) 
                      for _ in range(votes)]
            results = [f.result() for f in tqdm(futures, desc=f"Voting (N={votes})", leave=False)]

        all_p_yes = [r[0] for r in results]
        all_rationales = [r[1] for r in results]
        
        avg_p_yes = np.mean(all_p_yes)
        final_verdict = avg_p_yes > self.confidence_threshold
        
        return final_verdict, all_rationales, avg_p_yes

    def evaluate_dataset(self, dataset, sample_size, votes):
        """Full evaluation pipeline with progress tracking"""
        results = []
        sample = random.sample(dataset, min(sample_size, len(dataset)))
        
        for idx, item in enumerate(tqdm(sample, desc="GenRM-CoT Evaluation"), 1):
            prompt = str(item['prompt'])
            chosen = str(item['chosen'][0] if isinstance(item['chosen'], list) else item['chosen'])
            rejected = str(item['rejected'][0] if isinstance(item['rejected'], list) else item['rejected'])
            domain = str(item.get('domain', 'unknown'))

            # Evaluate chosen answer
            chosen_verdict, chosen_rationales, chosen_score = self.verify_cot_majority(prompt, chosen, votes)
            
            # Evaluate rejected answer
            _, rejected_rationales, rejected_score = self.verify_cot_majority(prompt, rejected, votes)
            
            result = {
                'type': 'cot',
                'votes': votes,
                'chosen_score': chosen_score,
                'rejected_score': rejected_score,
                'verdict': chosen_verdict,
                'sample_rationale': chosen_rationales[0] if chosen_rationales else None,
                'rejected_sample_rationale': rejected_rationales[0] if rejected_rationales else None
            }

            passed = chosen_verdict and (rejected_score < self.confidence_threshold)
            results.append({
                'id': idx,
                'prompt': prompt,
                'domain': domain,
                'passed': passed,
                **result
            })

            color = Fore.GREEN if passed else Fore.RED
            print(f"{color}Item {idx}: {'✓' if passed else '✗'} | "
                  f"Chosen: {chosen_score:.2%} | Rejected: {rejected_score:.2%} | Domain: {domain}")


        accuracy = sum(r['passed'] for r in results) / len(results) if results else 0.0
        print(f"\n{Fore.BLUE}GENRM-CoT RESULTS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Accuracy:{Style.RESET_ALL} {accuracy:.2%}")
        print(f"{Fore.YELLOW}Avg Chosen Score:{Style.RESET_ALL} {np.mean([r['chosen_score'] for r in results]):.2%}")
        print(f"{Fore.YELLOW}Avg Rejected Score:{Style.RESET_ALL} {np.mean([r['rejected_score'] for r in results]):.2%}")
        return accuracy, results

    def save_results(self, results):
      """Structured results saving with metadata"""
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      output = {
         "metadata": {
               "evaluation_date": timestamp,
               "model": self.model,
               "confidence_threshold": self.confidence_threshold,
               "total_samples": len(results),
               "passed_samples": sum(1 for r in results if bool(r['passed'])),  # Convert NumPy bool
               "accuracy": float(sum(1 for r in results if bool(r['passed']))) / len(results),
               "average_chosen_score": float(np.mean([r['chosen_score'] for r in results])),
               "average_rejected_score": float(np.mean([r['rejected_score'] for r in results]))
         },
         "results": [
               {**r, "passed": bool(r["passed"])}  # Ensure boolean conversion for each result
               for r in results
         ]
      }
      
      filename = f"genrm_results_{timestamp}.json"
      with open(filename, 'w') as f:
         json.dump(output, f, indent=2)
      
      print(f"\n{Fore.GREEN}Results saved to:{Style.RESET_ALL} {filename}")
      return filename


if __name__ == "__main__":
    # Initialize with your API key
    genrm = GenRM(api_key="")
    
    # Load dataset
    dataset = genrm.load_dataset(
        file_path="dataset/total_dataset.json",  
        sample_size=100
    )
    
    # Run evaluation with paper-specified parameters
    accuracy, results = genrm.evaluate_dataset(
        dataset, 
        sample_size=10, 
        votes=10
    )

    # Save structured results
    genrm.save_results(results)
