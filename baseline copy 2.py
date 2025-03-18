import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
import matplotlib.pyplot as plt
import json

# 1. CoT数据生成模块 (参照论文3.1节要求)
def generate_verification_cot(sampled_names, attempt, correct_answer):
    """Generate verification chain-of-thought (CoT)"""
    verification_steps = [
        f"Verification input: {' '.join(sampled_names)}",
        f"Generated answer: {attempt}",
        "Verification steps:",
        f"1. Check if the answer length is {len(sampled_names)} letters",
        f"2. Verify if each letter corresponds correctly:"
    ]
    
    for i, (name, char) in enumerate(zip(sampled_names, attempt)):
        verification_steps.append(
            f"   - The last letter of the {i+1}th word '{name}' should be '{name[-1]}', in the generated answer it is '{char}' → {'Correct' if char == name[-1] else 'Incorrect'}"
        )
    
    conclusion = "Correct" if attempt == correct_answer else "Incorrect"
    verification_steps.append(f"Final conclusion: Answer is {conclusion}")
    return "\n".join(verification_steps)

# 2. CoT数据生成模块 (GEMMA-2B生成的attempt仍有问题，目前手动添加correct_answer)
def generate_last_letter_data(dataset, n_query=350, lengths=[2,3,4], n_attempts=128, output_file="training_data.json"):
    all_data = []
    
    for length in lengths:
        for _ in range(n_query):
            # Randomly sample names
            sampled_names = np.random.choice(dataset, length, replace=False)
            correct_answer = ''.join([name[-1] for name in sampled_names])
            
            # Generate attempts
            attempts = []
            prompt = f"Take the last letters of each word in {' '.join(sampled_names)} and concatenate them.\nAnswer:"
            
            # Add the correct answer as the first attempt
            attempts.append({
                "attempt": correct_answer,
                "is_correct": True,
                "cot": generate_verification_cot(sampled_names, correct_answer, correct_answer)
            })
            
            # Generate incorrect attempts
            incorrect_attempts_needed = n_attempts // 2  # 50% incorrect attempts
            incorrect_attempts_generated = 0
            
            while incorrect_attempts_generated < incorrect_attempts_needed:
                # Generate additional attempts using the model
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=length,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.3,
                    num_return_sequences=1  # Generate one attempt at a time
                )
                
                # Decode and process the attempt
                attempt = tokenizer.decode(outputs[0], skip_special_tokens=True)
                attempt = attempt.replace(prompt, "").strip()
                
                # Only add the attempt if it's incorrect
                if attempt != correct_answer:
                    cot = generate_verification_cot(sampled_names, attempt, correct_answer)
                    attempts.append({
                        "attempt": attempt,
                        "is_correct": False,
                        "cot": cot
                    })
                    incorrect_attempts_generated += 1
            
            # Add the remaining correct attempts (if any)
            correct_attempts_needed = n_attempts - incorrect_attempts_generated
            for _ in range(correct_attempts_needed):
                attempts.append({
                    "attempt": correct_answer,
                    "is_correct": True,
                    "cot": generate_verification_cot(sampled_names, correct_answer, correct_answer)
                })
            
            # Shuffle attempts to mix correct and incorrect examples
            np.random.shuffle(attempts)
            
            all_data.append({
                "input": ' '.join(sampled_names),
                "correct_answer": correct_answer,
                "attempts": attempts
            })
    
    # Save data to JSON file
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=4)
    
    return all_data


# def generate_last_letter_data(dataset, n_query=350, lengths=[2,3,4], n_attempts=128, output_file="training_data.json"):
#     all_data = []
    
#     for length in lengths:
#         for _ in range(n_query):
#             # Randomly sample names
#             sampled_names = np.random.choice(dataset, length, replace=False)
#             correct_answer = ''.join([name[-1] for name in sampled_names])
            
#             # Generate attempts
#             attempts = []
#             prompt = f"Take the last letters of each words in {' '.join(sampled_names)} and concatenate them.\nAnswer:"
            
#             # Generate answers using the model
#             inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=length,
#                 do_sample=True,
#                 top_p=0.9,
#                 temperature=0.3,
#                 num_return_sequences=n_attempts
#             )
            
#             # Decode and process attempts
#             for output in outputs:
#                 attempt = tokenizer.decode(output, skip_special_tokens=True)
#                 attempt = attempt.replace(prompt, "").strip()
#                 cot = generate_verification_cot(sampled_names, attempt, correct_answer)
#                 attempts.append({
#                     "attempt": attempt,
#                     "is_correct": attempt == correct_answer,  # Fix typo here
#                     "cot": cot
#                 })
            
#             all_data.append({
#                 "input": ' '.join(sampled_names),
#                 "correct_answer": correct_answer,
#                 "attempts": attempts
#             })
    
#     # Save data to JSON file
#     with open(output_file, "w") as f:
#         json.dump(all_data, f, indent=4)
    
#     return all_data

# 3. 生成式验证器训练模块 (参照论文3.2节)
class VerifierDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for item in data:
            for attempt in item["attempts"]:
                # 构建生成式输入
                prompt = (
                    f"Problem: Connect the last letters of the following names: {item['input']}\n"
                    f"Candidate answer: {attempt['attempt']}\n"
                    f"Verification steps: {attempt['cot']}\n"
                    "Final judgment:"
                )
                target = "Correct" if attempt["is_correct"] else "Incorrect"
                
                # 编码样本
                encoded = tokenizer(
                    prompt + target,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                # 计算labels（mask掉输入部分）
                input_len = len(tokenizer(prompt)["input_ids"])
                labels = encoded["input_ids"].clone()
                labels[:, :input_len] = -100  # 忽略输入部分的loss
                
                self.examples.append({
                    "input_ids": encoded["input_ids"].squeeze(),
                    "attention_mask": encoded["attention_mask"].squeeze(),
                    "labels": labels.squeeze()
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# # 3. 准确率计算函数
# def compute_accuracy(eval_pred):
#     predictions, labels = eval_pred
#     # 获取预测结果中"正确"/"错误"的位置
#     preds_flat = np.argmax(predictions[0], axis=-1).flatten()
#     labels_flat = labels.flatten()
    
#     # 只计算有效标签位置（labels != -100）
#     mask = labels_flat != -100
#     valid_preds = preds_flat[mask]
#     valid_labels = labels_flat[mask]
    
#     # 解码token为文本
#     pred_text = tokenizer.batch_decode(valid_preds, skip_special_tokens=True)
#     label_text = tokenizer.batch_decode(valid_labels, skip_special_tokens=True)
    
#     # 计算匹配数量
#     correct = sum([1 for p, l in zip(pred_text, label_text) if p == l])
#     return {"accuracy": correct / len(pred_text)}

# 4. 增强准确率计算函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Flatten predictions and labels
    predictions = np.argmax(predictions, axis=-1).flatten()
    labels = labels.flatten()
    
    # Mask out invalid labels (where labels == -100)
    mask = labels != -100
    valid_preds = predictions[mask]
    valid_labels = labels[mask]
    
    # Decode predictions and labels to text
    pred_text = tokenizer.batch_decode(valid_preds, skip_special_tokens=True)
    label_text = tokenizer.batch_decode(valid_labels, skip_special_tokens=True)
    
    # Debugging: Print token IDs and their corresponding text
    # print(f"Predicted token IDs: {valid_preds}")
    # print(f"Label token IDs: {valid_labels}")
    # print(f"Predicted texts: {pred_text}")
    # print(f"Label texts: {label_text}")
    
    # Calculate accuracy
    correct = sum([1 for p, t in zip(pred_text, label_text) if p == t])
    accuracy = correct / len(pred_text) if len(pred_text) > 0 else 0.0
    return {"accuracy": round(accuracy, 4)}

# 5. 自定义回调函数记录训练指标
class MetricsLogger(TrainerCallback):
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'train_accuracy': [],
            'eval_accuracy': []
        }
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture training logs"""
        if logs:
            print(f"Logs received: {logs}")  # 打印日志内容
            if 'loss' in logs:
                self.metrics['train_loss'].append((state.epoch, logs['loss']))
            if 'accuracy' in logs:  # 检查是否有 accuracy
                self.metrics['train_accuracy'].append((state.epoch, logs['accuracy']))
                print(f"Training accuracy: {logs['accuracy']}") 
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Capture evaluation metrics"""
        if metrics:
            print(f"Evaluation metrics received: {metrics}")  # 打印评估指标
            self.metrics['eval_loss'].append((state.epoch, metrics['eval_loss']))
            if 'eval_accuracy' in metrics:  # 检查是否有 eval_accuracy
                self.metrics['eval_accuracy'].append((state.epoch, metrics['eval_accuracy']))
    
    def on_train_end(self, args, state, control, **kwargs):
        """Generate visualization after training"""
        self._plot_metrics()
    
    def _plot_metrics(self):
        plt.figure(figsize=(18, 6))
        
        # Plot Loss Curves
        plt.subplot(1, 3, 1)
        if self.metrics['train_loss']:
            train_epochs, train_loss = zip(*self.metrics['train_loss'])
            plt.plot(train_epochs, train_loss, label='Train Loss')
        if self.metrics['eval_loss']:
            eval_epochs, eval_loss = zip(*self.metrics['eval_loss'])
            plt.plot(eval_epochs, eval_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # # Plot Training Accuracy (if available)
        # plt.subplot(1, 3, 2)
        # if self.metrics['train_accuracy']:
        #     train_epochs, train_acc = zip(*self.metrics['train_accuracy'])
        #     plt.plot(train_epochs, train_acc, label='Train Accuracy', color='green')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.title('Training Accuracy')
        # plt.legend()
        
        # Plot Validation Accuracy
        plt.subplot(1, 3, 3)
        if self.metrics['eval_accuracy']:
            eval_epochs, eval_acc = zip(*self.metrics['eval_accuracy'])
            plt.plot(eval_epochs, eval_acc, label='Validation Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('./training_metrics.png')
        plt.close()

# 5. 推理模块改进（多数投票机制）
def majority_vote(model, prompt, num_samples=32, threshold=0.1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    yes_probs = []
    for _ in range(num_samples):
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=True,
            top_p=0.9,
            output_scores=True,
            return_dict_in_generate=True
        )
        logits = outputs.scores[0][0]
        yes_prob = torch.softmax(logits, dim=-1)[tokenizer("Correct").input_ids[0]].item()
        yes_probs.append(yes_prob)
    
    mean_yes_prob = np.mean(yes_probs)
    return "Correct" if mean_yes_prob > threshold else "Incorrect"

# # 5. 推理模块改进（多数投票机制）
# def majority_vote(model, prompt, num_samples=32):
#     # 多数投票推理（论文3.3节）
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     yes_probs = []
    
#     for _ in range(num_samples):
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=2,
#             do_sample=True,
#             top_p=0.9,
#             output_scores=True,
#             return_dict_in_generate=True
#         )
        
#         # 获取"Yes"/"No"的概率
#         logits = outputs.scores[0][0]
#         yes_prob = torch.softmax(logits, dim=-1)[tokenizer("正确").input_ids[0]].item()
#         yes_probs.append(yes_prob)
    
#     return np.mean(yes_probs)

# 使用示例
if __name__ == "__main__":

    # Gemma-2b模型加载
    model_path = "/root/autodl-tmp/Lucachen/gemma2b"
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(tokenizer.encode("Correct", add_special_tokens=False))  # 应输出单token如[28745]
    print(tokenizer.encode("Incorrect", add_special_tokens=False)) # 应输出单token如[30201]

    # 数据生成
    ln_dataset = pd.read_csv("top_1000_last_names.csv")['Name'].tolist()

    # print("start generating training data...")
    # train_data=generate_last_letter_data(ln_dataset, n_query=50, lengths=[2,3,4], n_attempts=20, output_file="training_data_manually.json")

    # print("start generating evaluation data...")
    # eval_data=generate_last_letter_data(ln_dataset, n_query=20, lengths=[6], n_attempts=2, output_file="eval_data_manually.json")
    
    train_data_file = "training_data_manually.json"
    eval_data_file = "eval_data_manually.json"
    with open(train_data_file, "r") as f:
        train_data = json.load(f)
    with open(eval_data_file, "r") as f:
        eval_data = json.load(f)

    # 准备数据集
    train_dataset = VerifierDataset(train_data, tokenizer)
    eval_dataset = VerifierDataset(eval_data, tokenizer)

    # 训练参数优化 
    training_args = TrainingArguments(
        output_dir="/root/autodl-tmp/gemma_verifier",
        evaluation_strategy="steps",
        eval_steps=20,
        save_strategy="no",
    #  save_steps=500,
        optim="adamw_torch_fused",
        learning_rate=2e-6,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        bf16=True,
        max_grad_norm=1.0,  # Gradient clipping
        lr_scheduler_type="cosine",  # Cosine decay
        warmup_steps=500,  # Linear warmup    
        gradient_checkpointing=False,
        logging_steps=20,
    )
  
    # 训练模型
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MetricsLogger()]  # 自定义回调
    )

    torch.cuda.empty_cache()

    trainer.train()
    
    # 保存模型
    # model.save_pretrained("/root/autodl-tmp/trained_verifier")
    # tokenizer.save_pretrained("/root/autodl-tmp/trained_verifier")