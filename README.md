# Offshoot-Model-Strategies-to-Assist-the-Reward-Model


## ```Dataset ``` ##

### Standard-format-preference-dataset ###

https://huggingface.co/collections/RLHFlow/standard-format-preference-dataset-662eec0252e194d5d40c252a

### GradeSchoolMath ###

dataset:https://github.com/openai/grade-school-math/tree/master/grade_school_math/data


### LastLetterConcat ### 

dataset:https://huggingface.co/datasets/ChilleD/LastLetterConcat/tree/main


### Word Sorting ### 

dataset:https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/word_sorting


### RM-Bench ### 

dataset:https://github.com/THU-KEG/RM-Bench/tree/main/data


| Dataset          | Total Samples |   Train/Test Split     | 
|------------------|---------------|------------------------|
| GradeSchoolMath  |     8.5K      |  7.5K Train / 1K Test  | 
| LastLetterConcat |     500       |  350 Train / 150 Test  | 
| Word Sorting     |     1.9K      |  Not splitted yet      |
| RM-Bench         |  1.3K (chat/code/math/safety)  |  Not splitted yet      |

## ```Stage1 data_generation_VerifyCoT ``` ##

VerifyCoT is particularly crucial for smaller models like Gemma-7B because they struggle with complex reasoning and logical consistency compared to larger models like GPT-3 or GPT-4. Unlike massive models that can often self-correct due to their extensive pretraining, smaller models require explicit verification to reduce errors and improve structured generation tasks.

Generating training data using a VerifyCoT (Verification Chain-of-Thought) approach. It leverages a pre-trained language model (Gemma-7B) to generate candidate answers, verify their correctness, and produce a dataset suitable for training or fine-tuning models. 


### ```step1 create_primary_prompt ``` ###

Very crucial part of guiding the language model to generate structured and valid outputs.

#### Step 1: Extract last letters ####

Add strong attention to the last letter with regular expression, as the original model intends to focus more on first letters.

#### Step 2: Convert to uppercase and remove non-alphabetic characters ####

#### Step 3: Concatenate the extracted letters ####

#### Step 4: Ensure Correct Length ####


### ```step2 Generate initial_answers ``` ###

Generates initial candidate answers using LLM(Gemma-7b).

```num_return_sequences=10``` : Currently generate 10 candidate answers.

```bad_words_ids=[[self.tokenizer.encode("<")[0]]]``` : Prevents the model from generating unwanted tokens (e.g., <), as original model intends to generate lots of html codes...


### ```step3 validate_and_correct ``` ###

KEY KEY KEY KEY part of the whole generation stage. 

#### Step 1: Verification ####

Validates a candidate answer using a verification prompt.

#### Step 2: Correction ####

Generate a correction based on the verification prompt.Enforce the model to generate the answers that meet the requirement.

```do_sample=False``` : Extremely important, greedy decoding, disable sampling, generate deterministic token with the highest probablity ONLY that meet the requirement of verification

#### Step 3: Decode the correction and extract the final answer ####



### ```step4 Generate initial_answers ``` ###

Generates multiple candidate answers, validates them, and selects the best one based on alignment with the ground truth.


As we are not encouraging diversity by adding randomness or do_sampling in the selection process. We currently deploy deterministic approach rather than probablistic (argmax rather than softmax)
