# Offshoot-Model-Strategies-to-Assist-the-Reward-Model


## ```Dataset ``` ##

### Standard-format-preference-dataset ###

https://huggingface.co/collections/RLHFlow/standard-format-preference-dataset-662eec0252e194d5d40c252a

### Dahoas Series### 

https://huggingface.co/Dahoas

### Dahoas/rm-static ### 

dataset:https://huggingface.co/datasets/Dahoas/rm-static


### RM-Bench ### 

dataset:https://github.com/THU-KEG/RM-Bench/tree/main/data


# RLHF Dataset Comparison

## Overview Table

| Dataset                 | Total Samples | Train/Test Split          | Best Use Case                               | Pros                                                                 | Cons                                                                 |
|-------------------------|---------------|---------------------------|---------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| **Dahoas/rm-static**    | 81K           | 76,256 Train / 5,103 Test | General chat alignment (customer service)  | ✅ Clean `chosen`/`rejected` pairs<br>✅ Multi-turn context support  | ❌ No technical depth<br>❌ Potential annotator bias                 |
| **Dahoas/filtered-SHP** | ~50K-100K     | Not split                 | Multi-topic preference learning            | ✅ Low noise (filtered)<br>✅ Multilingual support                  | ❌ English-centric bias<br>❌ Limited technical expertise            |
| **Dahoas/full-hh-rlhf** | 112K          | Not split                 | Safety & ethical alignment                  | ✅ Large-scale safety focus<br>✅ Complex multi-turn interactions  | ❌ Requires manual calibration<br>❌ Formatting adaptation needed    |
| **RM-Bench**            | 1.3K          | Not split                 | Code/math/safety evaluation                 | ✅ Technical domain coverage<br>✅ Granular difficulty levels       | ❌ Too small for training<br>❌ Style bias impacts accuracy          |

---

## Key Use Cases

### 1. **General Chatbots**
- **Best Datasets**: `rm-static` + `full-hh-rlhf`
- **Why**: Combines conversational alignment with safety guardrails.

### 2. **Technical Tasks (Code/Math)**
- **Evaluation**: Use `RM-Bench` for testing sensitivity to subtle errors.
- **Training**: Supplement with technical datasets (e.g., CodeXGLUE, GSM8K).

### 3. **Ethical AI Systems**
- **Best Dataset**: `full-hh-rlhf`
- **Note**: Validate with human reviewers to avoid overfitting to subjective preferences.

---


### Archive ### 

#### GradeSchoolMath ####

dataset:https://github.com/openai/grade-school-math/tree/master/grade_school_math/data


#### LastLetterConcat ####

dataset:https://huggingface.co/datasets/ChilleD/LastLetterConcat/tree/main


#### Word Sorting #### 

dataset:https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/word_sorting



| Dataset          | Total Samples |   Train/Test Split     | 
|------------------|---------------|------------------------|
| GradeSchoolMath  |     8.5K      |  7.5K Train / 1K Test  | 
| LastLetterConcat |     500       |  350 Train / 150 Test  | 
| Word Sorting     |     1.9K      |  Not splitted yet      |
