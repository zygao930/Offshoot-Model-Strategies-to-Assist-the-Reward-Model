import numpy as np
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

class VerifyCoTGenerator:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16  
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    # 1.very important. 
    # create primary prompt first, 
    # add STRONG ATTENTION TO THE LAST LETTER WITH REGULAR EXPRESSION, 
    # than clean and standardlize it.
    def create_primary_prompt(self, names):
        return f"""Follow these steps:
1. Extract last letters from: {' '.join(names)}{' | '.join([f'{n}→{n[-1].upper()}' for n in names])}
2. Convert to uppercase and remove non-alphabetic characters
3. Concatenate results maintaining word order
4. Ensure length matches {len(names)} characters
Final answer (uppercase letters only):"""
    
    # 2.Generate initial_answers 
    def generate_initial_answer(self, input_names):

        # Create the primary prompt for the model
        primary_prompt = self.create_primary_prompt(input_names)
        # Tokenize the prompt and move to the same device as the model
        inputs = self.tokenizer(primary_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate multiple sequences using the model
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=len(input_names)+4,  # Limit the number of new tokens to generate
            do_sample=True,  # Enable sampling for diverse outputs
            temperature=0.2,  # Control the randomness of predictions by scaling the logits
            top_p=0.85,  # Use nucleus sampling (top-p sampling)
            num_return_sequences=10,  # Generate 10 different sequences, also equals to 10 candidates
            # used to generate a lot of html code...bad code
            # Prevent the model from generating certain tokens
            bad_words_ids=[[self.tokenizer.encode("<")[0]]] 
        )
        
        # Decode the generated sequences into human-readable text
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


    # 3.Verification and correction. 
    #  REMOVE THE SYMBOLS->KEEP ALPHABETIC->STANDARDLIZE
    def validate_and_correct(self, attempt, input_names):
        expected_len=len(input_names)

        # Step1: ADD A VERIFICATION 
        verification_template = """Verify if the answer meets all requirements:
Input: {input_str}
Generated attempt: {attempt}
Verification steps:
1. Does each character match the last letter of corresponding input word?
2. Are all letters uppercase with no special characters?
3. Is the length exactly {expected_len}?
Corrected answer:"""

        # Format the verification prompt with the input, attempt, and expected length
        verification_prompt = verification_template.format(
            input_str=' '.join(input_names),
            attempt=attempt,
            expected_len=expected_len
        )
        
        inputs = self.tokenizer(verification_prompt, return_tensors="pt").to(self.model.device)

        # Step2: Generate a correction based on the verification prompt
        # Enforce the model to generate the answers that meet the requirement
        correction = self.model.generate(
            **inputs,
            max_new_tokens=expected_len+2,  # Limit the number of new tokens

            # Extremely important, 
            # greedy decoding, disable sampling, generate deterministic token with the highest probablity
            # that meet the requirement of verification
            do_sample=False  
        )
        
        # Decode the correction and extract the final answer
        final_answer = self.tokenizer.decode(correction[0], skip_special_tokens=True)
        return re.sub(r'[^A-Z]', '', final_answer.split("Corrected answer:")[-1].strip())

    #4.Generate the best answer using Verify-CoT generation with a validation loop.
    def get_best_answer(self, input_names):

        # Generate initial candidate answers
        candidates = self.generate_initial_answer(input_names)
        # Validate and correct each candidate
        validated = [self.validate_and_correct(a, input_names) for a in candidates]
        
        # Calculate the ground truth for comparison
        ground_truth = ''.join([n[-1].upper() for n in input_names])
        # Score each validated answer based on how many characters match the ground truth
        scores = [sum(gt == pred for gt, pred in zip(ground_truth, a)) for a in validated]
        # Return the answer with the highest score
        return validated[np.argmax(scores)]

class TrainingDataGenerator:
    def __init__(self, model_path):
        self.generator = VerifyCoTGenerator(model_path)
    
    def create_dataset(self, name_list, config):
        # Generate multi-verification training data.
        dataset = []
        # Iterate over different lengths specified in the config
        for length in config['lengths']:
            # Generate a specified number of samples for each length
            for _ in range(config['n_samples']):
                # Randomly select names from the list
                names = np.random.choice(name_list, length, replace=False)
                # Calculate the ground truth by taking the last letter of each name
                ground_truth = ''.join([n[-1].upper() for n in names])
                
                attempts = []
                # Generate multiple attempts for each sample
                for _ in range(config['n_attempts']):
                    # Get the best answer using the MetaCoTGenerator
                    attempt = self.generator.get_best_answer(names)
                    # Check if the attempt matches the ground truth
                    is_correct = (attempt == ground_truth)
                    
                    # Create a verification chain for the attempt
                    verification_chain = self._create_verification_chain(
                        names, attempt, ground_truth
                    )
                    # Store the attempt along with its correctness and verification steps
                    attempts.append({
                        "attempt": attempt,
                        "is_correct": is_correct,
                        "verification_steps": verification_chain
                    })
                
                # Store the input sequence, ground truth, and attempts in the dataset
                dataset.append({
                    "input_sequence": ' '.join(names),
                    "ground_truth": ground_truth,
                    "attempts": attempts
                })
        return dataset

    def _create_verification_chain(self, names, attempt, ground_truth):
        verification_steps = [
            f"Input: {' '.join(names)}",
            f"Attempt: {attempt}",
            f"Expected: {ground_truth}",
            "Verification Process:"
        ]
        
        # Compare each character in the attempt with the expected character
        # Very interesting point here "✓✗" is more distinguisble, Unicode characters
        # right/wrong can also contains ethics judgment, while "✓✗" focuses more on correct/incorrect, but with less space
        for idx, (name, char) in enumerate(zip(names, attempt)):
            expected_char = name[-1].upper()
            status = "✓" if char == expected_char else f"✗ (Expected: {expected_char})"
            verification_steps.append(
                f"{idx+1}. {name} → {char} {status}"
            )
        
        # Conclude whether the attempt is valid or invalid
        conclusion = "Valid" if attempt == ground_truth else "Invalid"
        verification_steps.append(f"Final Result: {conclusion}")
        return "\n".join(verification_steps)

if __name__ == "__main__":
    model_path = "/root/autodl-tmp/LLM-Research/gemma-7b"
    data_generator = TrainingDataGenerator(model_path)
    
    train_config = {
        'lengths': [2, 3, 4],  
        'n_samples': 5,  
        'n_attempts': 5  
    }
    
    last_names = pd.read_csv("/root/dl/top_1000_last_names.csv")['Name'].tolist()
    training_data = data_generator.create_dataset(last_names, train_config)
    with open("/root/dl/enhanced_training_data.json", "w") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
