import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner.trainer import train
from mlx_lm.tuner.utils import build_schedule
import json
import argparse
from pathlib import Path

# Configuration for TOEFL judge fine-tuning
TRAINING_CONFIG = {
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",  # Use 4-bit quantized version for efficiency
    "train_path": "toefl_speaking_data_train.jsonl",
    "val_path": "toefl_speaking_data_val.jsonl", 
    "adapter_path": "toefl_judge_adapter",
    "save_every": 100,
    "steps": 1000,
    "learning_rate": 1e-5,
    "batch_size": 4,
    "lora_layers": 16,
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "warmup_steps": 100,
    "max_seq_length": 2048,
    "grad_checkpoint": True
}

def setup_training_config():
    """Setup the configuration file for MLX training"""
    config = {
        "lora_layers": TRAINING_CONFIG["lora_layers"],
        "lora_rank": TRAINING_CONFIG["lora_rank"], 
        "lora_alpha": TRAINING_CONFIG["lora_alpha"],
        "lora_dropout": TRAINING_CONFIG["lora_dropout"]
    }
    
    with open("lora_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return "lora_config.json"

def validate_data_format(jsonl_path):
    """Validate that the JSONL data is properly formatted"""
    print(f"Validating data format in {jsonl_path}...")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                
                # Check required structure
                assert "messages" in data, f"Line {i+1}: Missing 'messages' key"
                messages = data["messages"]
                assert len(messages) == 3, f"Line {i+1}: Expected 3 messages (system, user, assistant)"
                
                # Check message roles
                assert messages[0]["role"] == "system", f"Line {i+1}: First message should be system"
                assert messages[1]["role"] == "user", f"Line {i+1}: Second message should be user"
                assert messages[2]["role"] == "assistant", f"Line {i+1}: Third message should be assistant"
                
                # Check content exists
                for j, msg in enumerate(messages):
                    assert "content" in msg and msg["content"].strip(), f"Line {i+1}, Message {j+1}: Empty content"
                
                if i < 3:  # Show first few examples
                    print(f"Example {i+1} validated successfully")
                    
            except Exception as e:
                print(f"Error in line {i+1}: {e}")
                return False
                
            if i >= 10:  # Check first 10 lines
                break
    
    print(f"Data format validation passed!")
    return True

def run_training():
    """Execute the MLX fine-tuning process"""
    
    # Validate data first
    if not validate_data_format(TRAINING_CONFIG["train_path"]):
        print("Training data validation failed!")
        return
    
    if not validate_data_format(TRAINING_CONFIG["val_path"]):
        print("Validation data validation failed!")
        return
    
    # Setup LoRA config
    config_path = setup_training_config()
    
    print("Starting TOEFL judge fine-tuning...")
    print(f"Model: {TRAINING_CONFIG['model']}")
    print(f"Training steps: {TRAINING_CONFIG['steps']}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    
    # MLX training command (you would run this via command line)
    training_command = f"""
mlx_lm.lora \\
    --model {TRAINING_CONFIG['model']} \\
    --train \\
    --data {TRAINING_CONFIG['train_path']} \\
    --val-data {TRAINING_CONFIG['val_path']} \\
    --adapter-path {TRAINING_CONFIG['adapter_path']} \\
    --iters {TRAINING_CONFIG['steps']} \\
    --learning-rate {TRAINING_CONFIG['learning_rate']} \\
    --batch-size {TRAINING_CONFIG['batch_size']} \\
    --lora-layers {TRAINING_CONFIG['lora_layers']} \\
    --lora-rank {TRAINING_CONFIG['lora_rank']} \\
    --lora-alpha {TRAINING_CONFIG['lora_alpha']} \\
    --lora-dropout {TRAINING_CONFIG['lora_dropout']} \\
    --save-every {TRAINING_CONFIG['save_every']} \\
    --max-tokens {TRAINING_CONFIG['max_seq_length']} \\
    --grad-checkpoint
"""
    
    print("Run this command to start training:")
    print(training_command)
    
    # Save command to file for easy execution
    with open("run_training.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(training_command)
    
    print("\nTraining command saved to run_training.sh")
    print("Make it executable with: chmod +x run_training.sh")
    print("Then run: ./run_training.sh")

def test_model_inference():
    """Test the fine-tuned model with sample TOEFL questions"""
    
    # Load the fine-tuned model
    model_path = TRAINING_CONFIG['model']
    adapter_path = TRAINING_CONFIG['adapter_path']
    
    print(f"Loading model from {model_path} with adapter {adapter_path}")
    
    # Sample test cases
    test_cases = [
        {
            "question": "Do you agree or disagree: social media has a positive impact on society?",
            "answer": "I believe social media has both positive and negative impacts, but overall more positive. It connects people across distances, allows sharing of important information quickly, and gives voice to marginalized communities. However, it also spreads misinformation and can be addictive. The key is using it responsibly and being critical of information we see."
        },
        {
            "question": "Would you prefer to study abroad or in your home country?", 
            "answer": "Study abroad is better I think. You can learn new culture and improve language skills. Also meet many international students. But it expensive and far from family."
        }
    ]
    
    system_prompt = """You are an expert TOEFL iBT Independent Speaking evaluator. Your task is to score student responses on a scale of 0-4 based on the official TOEFL Independent Speaking rubric.

Scoring Criteria:
- Score 4: Fulfills task demands with sustained, coherent discourse. Well-paced delivery, effective grammar/vocabulary, well-developed and coherent ideas.
- Score 3: Addresses task appropriately but may lack full development. Generally clear speech with some fluency, fairly effective language use, mostly coherent with some limitations.
- Score 2: Addresses task but limited development. Intelligible speech requiring listener effort, limited grammar/vocabulary range, basic ideas with limited elaboration.
- Score 1: Very limited content/coherence, largely unintelligible speech, severely limited language control, lacks substance beyond basic ideas.
- Score 0: No attempt or unrelated to topic.

Provide your score and a brief explanation focusing on delivery, language use, and topic development."""
    
    # Code for inference (to be run after training)
    inference_code = '''
# After training, use this code to test your model:

from mlx_lm import load, generate

# Load the fine-tuned model
model, tokenizer = load(
    path_or_hf_repo="''' + model_path + '''",
    adapter_path="''' + adapter_path + '''"
)

# Test with sample cases
for i, test_case in enumerate(test_cases):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
Please evaluate this TOEFL Independent Speaking response:

Question: {test_case['question']}

Student Response: {test_case['answer']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt,
        max_tokens=300,
        temp=0.3
    )
    
    print(f"\\nTest Case {i+1}:")
    print(f"Question: {test_case['question']}")
    print(f"Answer: {test_case['answer']}")
    print(f"Model Evaluation: {response}")
    print("-" * 80)
'''
    
    with open("test_inference.py", "w") as f:
        f.write(inference_code)
    
    print("Inference test code saved to test_inference.py")
    print("Run this after training completes to test your model")

if __name__ == "__main__":
    print("TOEFL Judge MLX Fine-tuning Setup")
    print("="*50)
    
    # Check if data files exist
    train_path = Path(TRAINING_CONFIG["train_path"])
    val_path = Path(TRAINING_CONFIG["val_path"])
    
    if not train_path.exists():
        print(f"Error: Training data file {train_path} not found!")
        print("Please run the data formatter script first to generate the JSONL files.")
        exit(1)
    
    if not val_path.exists():
        print(f"Error: Validation data file {val_path} not found!")
        print("Please run the data formatter script first to generate the JSONL files.")
        exit(1)
    
    # Setup training
    run_training()
    
    # Generate test inference code
    test_model_inference()
    
    print("\nSetup complete! Next steps:")
    print("1. Run: chmod +x run_training.sh")
    print("2. Execute: ./run_training.sh")
    print("3. After training, test with: python test_inference.py")