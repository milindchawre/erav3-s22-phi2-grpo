# Phi-2 Model Training with GRPO and QLoRA

This project implements fine-tuning of the Microsoft Phi-2 model using GRPO (Group Relative Policy Optimization) and QLoRA (Quantized Low-Rank Adaptation) on the TLDR dataset. The training process focuses on optimizing the model's behavior relative to a reference group for generating concise and accurate text summaries.

## Features

- Fine-tuning Microsoft Phi-2 model using GRPO
- QLoRA implementation for efficient training
- Custom reward function for accuracy and semantic similarity
- TLDR dataset integration
- Optimized training configuration
- Real-time training metrics visualization

## Training Metrics

During training, the following metrics are tracked and visualized using TensorBoard:

- Reward: The combined reward value based on accuracy and semantic similarity
- Reward Standard Deviation: Variation in reward values across the batch
- KL Divergence: Measure of difference between model's output and reference distributions
- Loss: Training loss value

## Viewing Training Metrics

To view the training metrics in real-time:

1. Install TensorBoard if not already installed:
   ```bash
   pip install tensorboard
   ```

2. Start TensorBoard server:
   ```bash
   tensorboard --logdir=phi2-tldr-grpo/logs
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:6006
   ```

The metrics will be updated in real-time as the training progresses.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Training Process

The training script (`train.py`) implements:

1. QLoRA configuration for efficient fine-tuning
2. Custom reward function evaluating:
   - Accuracy through token overlap
   - Semantic similarity through cosine similarity
3. GRPO training setup with group-based policy optimization
4. Dataset preparation and formatting

## Usage

To start the training process:

```bash
python train.py
```

The trained model and checkpoints will be saved in the `phi2-tldr-grpo` directory.

## Model Configuration

- Base Model: microsoft/phi-2
- Training Method: Group Relative Policy Optimization (GRPO) with QLoRA
- Batch Size: 4
- Learning Rate: 1e-4
- Training Epochs: 3
- LoRA Configuration:
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.1

## Reward Function

The custom reward function evaluates model outputs based on:
- 70% weight on accuracy (token overlap with reference)
- 30% weight on semantic similarity (cosine similarity between embeddings)

## Output

The model generates concise summaries while maintaining accuracy with respect to the input text. Checkpoints are saved after each epoch in the output directory.