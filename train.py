import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from typing import Dict, List
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

# Configure model and training parameters
MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "phi2-tldr-grpo"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_STEPS = 500
SAVE_STEPS = 100
LOGGING_STEPS = 10

# Configure QLoRA parameters
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_R = 8
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "dense"]

def create_reward_fn(tokenizer):
    """Create a reward function that evaluates accuracy and semantic similarity"""
    def compute_reward(predictions: List[str], references: List[str]) -> Dict[str, torch.Tensor]:
        rewards = []
        for pred, ref in zip(predictions, references):
            # Get embeddings for prediction and reference
            pred_inputs = tokenizer(pred, return_tensors="pt", padding=True, truncation=True)
            ref_inputs = tokenizer(ref, return_tensors="pt", padding=True, truncation=True)
            
            # Convert token IDs to one-hot encodings for cosine similarity
            pred_one_hot = F.one_hot(pred_inputs["input_ids"].squeeze(), num_classes=tokenizer.vocab_size).float()
            ref_one_hot = F.one_hot(ref_inputs["input_ids"].squeeze(), num_classes=tokenizer.vocab_size).float()
            
            # Calculate cosine similarity
            pred_embed = pred_one_hot.mean(dim=0)
            ref_embed = ref_one_hot.mean(dim=0)
            cosine_sim = F.cosine_similarity(pred_embed.unsqueeze(0), ref_embed.unsqueeze(0))
            
            # Calculate accuracy using token overlap
            pred_set = set(pred_inputs["input_ids"].squeeze().tolist())
            ref_set = set(ref_inputs["input_ids"].squeeze().tolist())
            overlap = len(pred_set.intersection(ref_set))
            accuracy = overlap / max(len(pred_set), len(ref_set))
            
            # Combine accuracy and semantic similarity
            reward = (0.7 * accuracy + 0.3 * cosine_sim)
            rewards.append(reward)
        
        rewards_tensor = torch.tensor(rewards)
        return {
            "reward": rewards_tensor,
            "reward_std": rewards_tensor.std()
        }
    
    return compute_reward

def prepare_dataset(dataset):
    """Prepare the TLDR dataset for training"""
    def format_data(example):
        return {
            "input": f"TEXT: {example['prompt']}\n\nTL;DR:",
            "output": example['completion']
        }
    
    return dataset.map(format_data)

def main():
    # Load tokenizer and configure padding
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
        add_eos_token=True
    )
    # Add and configure padding token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'
    
    # Configure quantization for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Resize token embeddings after model initialization
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Load and prepare dataset
    dataset = load_dataset("trl-lib/tldr")
    train_dataset = prepare_dataset(dataset["train"])
    
    # Configure GRPO training parameters
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        remove_unused_columns=False,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        optim="paged_adamw_32bit",
        report_to=["tensorboard"],
        logging_first_step=True,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        max_prompt_length=512,
        max_completion_length=128,
        temperature=0.7,
        num_generations=4
    )
    
    # Create reward function
    reward_fn = create_reward_fn(tokenizer)
    
    # Custom callback to log metrics
    class MetricsCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.writer = SummaryWriter(os.path.join(OUTPUT_DIR, "logs"))

        def on_step_end(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                # Log reward metrics
                if "reward" in logs:
                    self.writer.add_scalar("train/reward", logs["reward"], state.global_step)
                if "reward_std" in logs:
                    self.writer.add_scalar("train/reward_std", logs["reward_std"], state.global_step)
                # Log KL divergence
                if "kl" in logs:
                    self.writer.add_scalar("train/kl", logs["kl"], state.global_step)
                # Log loss
                if "loss" in logs:
                    self.writer.add_scalar("train/loss", logs["loss"], state.global_step)

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        reward_funcs=reward_fn,
        callbacks=[MetricsCallback()]
    )

    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model()

if __name__ == "__main__":
    main()