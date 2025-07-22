import json
import pandas as pd
import numpy as np
from pathlib import Path
from mlx_lm import load, generate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import re
from collections import defaultdict

class TOEFLModelEvaluator:
    def __init__(self, model_path, adapter_path, test_data_path):
        """
        Initialize the TOEFL model evaluator
        
        Args:
            model_path: Path to the base model
            adapter_path: Path to the fine-tuned adapter
            test_data_path: Path to test data (CSV or JSONL)
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.test_data_path = test_data_path
        self.model = None
        self.tokenizer = None
        self.test_data = []
        self.predictions = []
        self.results = {}
        
    def load_model(self):
        """Load the fine-tuned model"""
        print("Loading fine-tuned model...")
        try:
            self.model, self.tokenizer = load(
                path_or_hf_repo=self.model_path,
                adapter_path=self.adapter_path
            )
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_test_data(self):
        """Load test data from CSV or JSONL format"""
        print(f"Loading test data from {self.test_data_path}...")
        
        if self.test_data_path.endswith('.csv'):
            df = pd.read_csv(self.test_data_path)
            self.test_data = [
                {
                    'question': row['question'].strip('"'),
                    'answer': row['answer'].strip('"'),
                    'true_score': int(row['score'])
                }
                for _, row in df.iterrows()
            ]
        elif self.test_data_path.endswith('.jsonl'):
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    # Extract from conversation format
                    user_content = data['messages'][1]['content']
                    assistant_content = data['messages'][2]['content']
                    
                    # Parse question and answer from user content
                    question_match = re.search(r'Question: (.+?)\n\nStudent Response:', user_content, re.DOTALL)
                    answer_match = re.search(r'Student Response: (.+)', user_content, re.DOTALL)
                    
                    # Parse true score from assistant content
                    score_match = re.search(r'Score: (\d+)', assistant_content)
                    
                    if question_match and answer_match and score_match:
                        self.test_data.append({
                            'question': question_match.group(1).strip(),
                            'answer': answer_match.group(1).strip(),
                            'true_score': int(score_match.group(1))
                        })
        
        print(f"Loaded {len(self.test_data)} test examples")
        return len(self.test_data) > 0
    
    def create_prompt(self, question, answer):
        """Create evaluation prompt for the model"""
        system_prompt = """You are an expert TOEFL iBT Independent Speaking evaluator. Your task is to score student responses on a scale of 0-4 based on the official TOEFL Independent Speaking rubric.

Scoring Criteria:
- Score 4: Fulfills task demands with sustained, coherent discourse. Well-paced delivery, effective grammar/vocabulary, well-developed and coherent ideas.
- Score 3: Addresses task appropriately but may lack full development. Generally clear speech with some fluency, fairly effective language use, mostly coherent with some limitations.
- Score 2: Addresses task but limited development. Intelligible speech requiring listener effort, limited grammar/vocabulary range, basic ideas with limited elaboration.
- Score 1: Very limited content/coherence, largely unintelligible speech, severely limited language control, lacks substance beyond basic ideas.
- Score 0: No attempt or unrelated to topic.

Provide your score and a brief explanation focusing on delivery, language use, and topic development."""
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
Please evaluate this TOEFL Independent Speaking response:

Question: {question}

Student Response: {answer}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return prompt
    
    def extract_score_from_response(self, response):
        """Extract numerical score from model response"""
        # Look for "Score: X" pattern
        score_match = re.search(r'Score:\s*(\d+)', response)
        if score_match:
            return int(score_match.group(1))
        
        # Look for standalone number at beginning
        number_match = re.search(r'^\s*(\d+)', response)
        if number_match:
            score = int(number_match.group(1))
            if 0 <= score <= 4:
                return score
        
        # If no clear score found, return None
        return None
    
    def evaluate_sample(self, question, answer, max_tokens=300, temp=0.1):
        """Evaluate a single sample"""
        prompt = self.create_prompt(question, answer)
        
        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens
            )
            
            predicted_score = self.extract_score_from_response(response)
            
            return {
                'response': response,
                'predicted_score': predicted_score
            }
        except Exception as e:
            print(f"Error evaluating sample: {e}")
            return {
                'response': f"Error: {e}",
                'predicted_score': None
            }
    
    def run_evaluation(self, max_samples=None, save_predictions=True):
        """Run evaluation on test dataset"""
        if not self.model or not self.tokenizer:
            print("Model not loaded! Please call load_model() first.")
            return
        
        if not self.test_data:
            print("Test data not loaded! Please call load_test_data() first.")
            return
        
        # Limit samples if specified
        test_samples = self.test_data[:max_samples] if max_samples else self.test_data
        
        print(f"Evaluating {len(test_samples)} samples...")
        
        self.predictions = []
        valid_predictions = []
        true_scores = []
        
        for i, sample in enumerate(test_samples):
            print(f"Evaluating sample {i+1}/{len(test_samples)}", end='\r')
            
            result = self.evaluate_sample(sample['question'], sample['answer'])
            
            prediction_data = {
                'question': sample['question'],
                'answer': sample['answer'],
                'true_score': sample['true_score'],
                'predicted_score': result['predicted_score'],
                'model_response': result['response']
            }
            
            self.predictions.append(prediction_data)
            
            # Only include valid predictions in metrics
            if result['predicted_score'] is not None:
                valid_predictions.append(result['predicted_score'])
                true_scores.append(sample['true_score'])
        
        print(f"\nCompleted evaluation!")
        print(f"Valid predictions: {len(valid_predictions)}/{len(test_samples)}")
        
        # Calculate metrics
        if len(valid_predictions) > 0:
            self.calculate_metrics(true_scores, valid_predictions)
        
        # Save predictions
        if save_predictions:
            self.save_predictions()
        
        return self.results
    
    def calculate_metrics(self, true_scores, predicted_scores):
        """Calculate evaluation metrics"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Basic accuracy
        exact_accuracy = accuracy_score(true_scores, predicted_scores)
        print(f"Exact Score Accuracy: {exact_accuracy:.3f}")
        
        # Adjacency accuracy (within 1 point)
        adjacent_correct = sum(1 for t, p in zip(true_scores, predicted_scores) 
                              if abs(t - p) <= 1)
        adjacent_accuracy = adjacent_correct / len(true_scores)
        print(f"Adjacent Accuracy (±1): {adjacent_accuracy:.3f}")
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(true_scores, predicted_scores)
        spearman_r, spearman_p = spearmanr(true_scores, predicted_scores)
        print(f"Pearson Correlation: {pearson_r:.3f} (p={pearson_p:.3f})")
        print(f"Spearman Correlation: {spearman_r:.3f} (p={spearman_p:.3f})")
        
        # Mean Absolute Error
        mae = np.mean([abs(t - p) for t, p in zip(true_scores, predicted_scores)])
        print(f"Mean Absolute Error: {mae:.3f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_scores, predicted_scores, 
                                  target_names=[f'Score {i}' for i in range(1, 5)]))
        
        # Store results
        self.results = {
            'exact_accuracy': exact_accuracy,
            'adjacent_accuracy': adjacent_accuracy,
            'pearson_correlation': pearson_r,
            'spearman_correlation': spearman_r,
            'mae': mae,
            'true_scores': true_scores,
            'predicted_scores': predicted_scores
        }
        
        return self.results
    
    def plot_results(self):
        """Create visualization plots"""
        if not self.results:
            print("No results to plot! Run evaluation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(self.results['true_scores'], 
                            self.results['predicted_scores'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], 
                   xticklabels=[1,2,3,4], yticklabels=[1,2,3,4])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted Score')
        axes[0,0].set_ylabel('True Score')
        
        # Score Distribution Comparison
        x = np.arange(1, 5)
        true_dist = [self.results['true_scores'].count(i) for i in range(1, 5)]
        pred_dist = [self.results['predicted_scores'].count(i) for i in range(1, 5)]
        
        width = 0.35
        axes[0,1].bar(x - width/2, true_dist, width, label='True Scores', alpha=0.7)
        axes[0,1].bar(x + width/2, pred_dist, width, label='Predicted Scores', alpha=0.7)
        axes[0,1].set_title('Score Distribution')
        axes[0,1].set_xlabel('Score')
        axes[0,1].set_ylabel('Count')
        axes[0,1].legend()
        axes[0,1].set_xticks(x)
        
        # Scatter plot
        axes[1,0].scatter(self.results['true_scores'], 
                         self.results['predicted_scores'], alpha=0.6)
        axes[1,0].plot([1, 4], [1, 4], 'r--', alpha=0.8)
        axes[1,0].set_title('True vs Predicted Scores')
        axes[1,0].set_xlabel('True Score')
        axes[1,0].set_ylabel('Predicted Score')
        axes[1,0].set_xlim(0.5, 4.5)
        axes[1,0].set_ylim(0.5, 4.5)
        
        # Error distribution
        errors = [p - t for t, p in zip(self.results['true_scores'], 
                                       self.results['predicted_scores'])]
        axes[1,1].hist(errors, bins=np.arange(-3.5, 4.5, 1), alpha=0.7)
        axes[1,1].set_title('Prediction Error Distribution')
        axes[1,1].set_xlabel('Prediction Error (Predicted - True)')
        axes[1,1].set_ylabel('Count')
        axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('toefl_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Evaluation plots saved as 'toefl_evaluation_results.png'")
    
    def save_predictions(self, filename='toefl_predictions.csv'):
        """Save detailed predictions to CSV"""
        if not self.predictions:
            print("No predictions to save!")
            return
        
        df = pd.DataFrame(self.predictions)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Predictions saved to {filename}")
    
    def analyze_errors(self, show_examples=5):
        """Analyze prediction errors in detail"""
        if not self.predictions:
            print("No predictions to analyze!")
            return
        
        # Find error cases
        errors = []
        for pred in self.predictions:
            if pred['predicted_score'] is not None:
                error = abs(pred['true_score'] - pred['predicted_score'])
                if error > 0:
                    errors.append({
                        'error_magnitude': error,
                        'true_score': pred['true_score'],
                        'predicted_score': pred['predicted_score'],
                        'question': pred['question'],
                        'answer': pred['answer'][:200] + '...',
                        'model_response': pred['model_response'][:300] + '...'
                    })
        
        # Sort by error magnitude
        errors.sort(key=lambda x: x['error_magnitude'], reverse=True)
        
        print(f"\nERROR ANALYSIS")
        print("="*50)
        print(f"Total errors: {len(errors)}")
        
        if len(errors) > 0:
            print(f"Average error magnitude: {np.mean([e['error_magnitude'] for e in errors]):.2f}")
            print(f"Max error magnitude: {max([e['error_magnitude'] for e in errors])}")
            
            print(f"\nTop {show_examples} error cases:")
            for i, error in enumerate(errors[:show_examples]):
                print(f"\n--- Error Case {i+1} ---")
                print(f"True Score: {error['true_score']}, Predicted: {error['predicted_score']}")
                print(f"Question: {error['question']}")
                print(f"Answer: {error['answer']}")
                print(f"Model Response: {error['model_response']}")
    
    def quick_test(self, question, answer):
        """Quick test with a single question-answer pair"""
        if not self.model:
            print("Model not loaded!")
            return
        
        result = self.evaluate_sample(question, answer)
        
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Model Response: {result['response']}")
        print(f"Extracted Score: {result['predicted_score']}")
        
        return result

# Usage example and main evaluation script
def main():
    """Main evaluation workflow"""
    
    # Configuration - UPDATE THESE PATHS
    MODEL_PATH = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    ADAPTER_PATH = "toefl_judge_adapter"  # Your trained adapter
    TEST_DATA_PATH = "toefl_speaking_data_test.jsonl"  # Or your test CSV
    
    # Initialize evaluator
    evaluator = TOEFLModelEvaluator(
        model_path=MODEL_PATH,
        adapter_path=ADAPTER_PATH,
        test_data_path=TEST_DATA_PATH
    )
    
    # Load model and data
    if not evaluator.load_model():
        print("Failed to load model!")
        return
    
    if not evaluator.load_test_data():
        print("Failed to load test data!")
        return
    
    # Quick test first
    print("Running quick test...")
    test_result = evaluator.quick_test(
        question="Do you prefer studying alone or in groups? Explain why.",
        answer="I prefer study in group because is more fun and you can help each other. When study alone is boring and if you don't understand something nobody can help. Group study better but sometimes noisy."
    )
    
    # Run full evaluation on subset first (to save time)
    print("\nRunning evaluation on 50 samples...")
    results = evaluator.run_evaluation(max_samples=50)
    
    if results:
        # Show detailed results
        evaluator.analyze_errors(show_examples=3)
        
        # Create plots
        evaluator.plot_results()
        
        # If results look good, run on full dataset
        user_input = input("\nRun evaluation on full dataset? (y/n): ")
        if user_input.lower() == 'y':
            print("Running full evaluation...")
            results = evaluator.run_evaluation()
            evaluator.plot_results()
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()