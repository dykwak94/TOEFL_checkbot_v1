import pandas as pd
import json
import random

def create_system_prompt():
    """Create the system prompt that defines the TOEFL judge role"""
    return """You are an expert TOEFL iBT Independent Speaking evaluator. Your task is to score student responses on a scale of 0-4 based on the official TOEFL Independent Speaking rubric.

Scoring Criteria:
- Score 4: Fulfills task demands with sustained, coherent discourse. Well-paced delivery, effective grammar/vocabulary, well-developed and coherent ideas.
- Score 3: Addresses task appropriately but may lack full development. Generally clear speech with some fluency, fairly effective language use, mostly coherent with some limitations.
- Score 2: Addresses task but limited development. Intelligible speech requiring listener effort, limited grammar/vocabulary range, basic ideas with limited elaboration.
- Score 1: Very limited content/coherence, largely unintelligible speech, severely limited language control, lacks substance beyond basic ideas.
- Score 0: No attempt or unrelated to topic.

Provide your score and a brief explanation focusing on delivery, language use, and topic development."""

def format_training_example(question, answer, score):
    """Format a single training example for MLX"""
    
    # Create detailed scoring rationale based on score level
    score_explanations = {
        1: f"Score: 1\n\nThis response demonstrates significant limitations. The answer shows very limited language control with frequent errors that impede communication. Topic development is minimal with basic ideas that lack proper elaboration. Delivery issues likely include pronunciation problems and choppy rhythm that require considerable listener effort.",
        
        2: f"Score: 2\n\nThis response addresses the task but with limited development. While basically intelligible, there are noticeable problems with delivery including unclear articulation or awkward pacing. Grammar and vocabulary range is limited, often preventing full expression of ideas. The response presents basic ideas but lacks sufficient detail and support.",
        
        3: f"Score: 3\n\nThis response appropriately addresses the task with generally clear speech and some fluency. Language use demonstrates fairly automatic and effective grammar and vocabulary, though there may be some imprecise usage that doesn't seriously interfere with communication. Topic development is mostly coherent but somewhat limited in elaboration or specificity.",
        
        4: f"Score: 4\n\nThis response fulfills the task demands excellently with sustained, coherent discourse. Delivery shows well-paced flow with clear speech and only minor lapses that don't affect intelligibility. Language use demonstrates effective grammar and vocabulary with good control of complex structures. Topic development is well-developed and coherent with clear progression of ideas."
    }
    
    return {
        "messages": [
            {
                "role": "system",
                "content": create_system_prompt()
            },
            {
                "role": "user", 
                "content": f"Please evaluate this TOEFL Independent Speaking response:\n\nQuestion: {question}\n\nStudent Response: {answer}"
            },
            {
                "role": "assistant",
                "content": score_explanations[score]
            }
        ]
    }

def convert_csv_to_jsonl(csv_file_path, output_file_path, train_split=0.6, val_split=0.2, test_split=0.2):
    """Convert CSV data to JSONL format for MLX fine-tuning with train/val/test split"""
    
    # Validate split ratios
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Clean and validate data
    df = df.dropna()  # Remove rows with missing values
    df = df[df['score'].isin([1, 2, 3, 4])]  # Keep only valid scores
    
    # Convert to training format
    training_data = []
    for _, row in df.iterrows():
        example = format_training_example(
            question=row['question'].strip('"'),
            answer=row['answer'].strip('"'), 
            score=int(row['score'])
        )
        training_data.append(example)
    
    # Shuffle the data
    random.shuffle(training_data)
    
    # Calculate split indices
    total_size = len(training_data)
    train_end = int(total_size * train_split)
    val_end = train_end + int(total_size * val_split)
    
    # Split into train, validation, and test sets
    train_data = training_data[:train_end]
    val_data = training_data[train_end:val_end]
    test_data = training_data[val_end:]
    
    # Write training data
    train_file = output_file_path.replace('.jsonl', '_train.jsonl')
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Write validation data  
    val_file = output_file_path.replace('.jsonl', '_val.jsonl')
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in val_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Write test data
    test_file = output_file_path.replace('.jsonl', '_test.jsonl')
    with open(test_file, 'w', encoding='utf-8') as f:
        for example in test_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Data split complete:")
    print(f"Training data: {len(train_data)} examples ({len(train_data)/total_size:.1%}) -> {train_file}")
    print(f"Validation data: {len(val_data)} examples ({len(val_data)/total_size:.1%}) -> {val_file}")
    print(f"Test data: {len(test_data)} examples ({len(test_data)/total_size:.1%}) -> {test_file}")
    
    return train_file, val_file, test_file

def create_sample_jsonl_examples():
    """Create sample JSONL examples to show the format"""
    examples = [
        {
            "question": "Do you agree or disagree with the following statement: it is never too late to get a degree in university?",
            "answer": "I think it never too late for university. My grandmother she go university when she 65 years old. She study history because she like it very much. Many older students in my country they very smart and work hard. Age not important for learning.",
            "score": 2
        },
        {
            "question": "Would you rather live in a house with a garden or an apartment in the city?",
            "answer": "I would prefer living in a house with a garden over a city apartment. Having private outdoor space provides invaluable benefits for mental health and lifestyle quality. A garden offers a peaceful retreat from daily stress, opportunities for physical activity through gardening, and the satisfaction of growing my own food. While city apartments offer convenience and social opportunities, the tranquility and connection to nature that a garden provides outweighs these advantages.",
            "score": 4
        }
    ]
    
    formatted_examples = []
    for ex in examples:
        formatted_examples.append(format_training_example(ex["question"], ex["answer"], ex["score"]))
    
    # Save sample examples
    with open('sample_training_examples.jsonl', 'w', encoding='utf-8') as f:
        for example in formatted_examples:
            f.write(json.dumps(example, ensure_ascii=False, indent=2) + '\n')
    
    print("Sample examples saved to sample_training_examples.jsonl")

# Usage example:
if __name__ == "__main__":
    # Create sample examples first
    create_sample_jsonl_examples()
    
    # Convert your CSV to JSONL format with 60/20/20 split
    train_file, val_file, test_file = convert_csv_to_jsonl(
        csv_file_path='Independent_Speaking_data.csv',
        output_file_path='toefl_speaking_data.jsonl',
        train_split=0.6,
        val_split=0.2,
        test_split=0.2
    )
    
    print(f"\nData conversion complete!")
    print(f"Use these files for MLX fine-tuning:")
    print(f"Training: {train_file}")
    print(f"Validation: {val_file}")
    print(f"Test: {test_file}")
    
    # Display data distribution by score for each split
    print("\nScore distribution across splits:")
    
    def analyze_split_distribution(file_path, split_name):
        scores = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                score_match = re.search(r'Score: (\d+)', data['messages'][2]['content'])
                if score_match:
                    scores.append(int(score_match.group(1)))
        
        score_counts = {i: scores.count(i) for i in range(1, 5)}
        print(f"{split_name}: Score 1: {score_counts[1]}, Score 2: {score_counts[2]}, Score 3: {score_counts[3]}, Score 4: {score_counts[4]}")
        return score_counts
    
    import re
    
    try:
        train_dist = analyze_split_distribution(train_file, "Train")
        val_dist = analyze_split_distribution(val_file, "Validation")
        test_dist = analyze_split_distribution(test_file, "Test")
    except Exception as e:
        print(f"Note: Could not analyze score distribution - {e}")