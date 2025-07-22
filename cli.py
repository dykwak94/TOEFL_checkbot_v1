#!/usr/bin/env python3
"""
Simple TOEFL Independent Speaking Judge - Command Line Interface
Usage: python toefl_judge_cli.py
"""

from mlx_lm import load, generate
import re
import argparse
import sys

class SimpleTOEFLJudge:
    def __init__(self, model_path="mlx-community/Llama-3.2-3B-Instruct-4bit", adapter_path="toefl_judge_adapter"):
        """Initialize the TOEFL judge with model paths"""
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
        print("🎓 TOEFL Independent Speaking Judge")
        print("="*50)
        print("Loading model... (this may take a moment)")
        
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model"""
        try:
            self.model, self.tokenizer = load(
                path_or_hf_repo=self.model_path,
                adapter_path=self.adapter_path
            )
            print("✅ Model loaded successfully!")
            print(f"📊 Model Performance: 86.1% accuracy, 0.943 correlation")
            print("="*50)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please check that your model and adapter paths are correct.")
            sys.exit(1)
    
    def create_prompt(self, question, answer):
        """Create the evaluation prompt"""
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
    
    def extract_score(self, response):
        """Extract numerical score from response"""
        score_match = re.search(r'Score:\s*(\d+)', response)
        return int(score_match.group(1)) if score_match else None
    
    def evaluate(self, question, answer):
        """Evaluate a TOEFL response"""
        print("🤔 Evaluating response...")
        
        try:
            prompt = self.create_prompt(question, answer)
            
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=400
            )
            
            score = self.extract_score(response)
            
            return {
                'score': score,
                'response': response,
                'success': True
            }
            
        except Exception as e:
            return {
                'score': None,
                'response': f"Error: {e}",
                'success': False
            }
    
    def print_result(self, question, answer, result):
        """Print evaluation results in a nice format"""
        print("\n" + "="*60)
        print("📝 EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n💬 Question:")
        print(f"{question}")
        
        print(f"\n🗣️  Student Response:")
        print(f"{answer}")
        
        if result['success'] and result['score'] is not None:
            score = result['score']
            
            # Score with emoji
            if score == 4:
                print(f"\n🌟 SCORE: {score}/4 (Excellent)")
            elif score == 3:
                print(f"\n✅ SCORE: {score}/4 (Good)")
            elif score == 2:
                print(f"\n⚠️  SCORE: {score}/4 (Limited)")
            elif score == 1:
                print(f"\n❌ SCORE: {score}/4 (Very Limited)")
            else:
                print(f"\n💀 SCORE: {score}/4 (No Response)")
            
            print(f"\n📋 Detailed Feedback:")
            print(f"{result['response']}")
            
        else:
            print(f"\n❌ Evaluation failed:")
            print(f"{result['response']}")
        
        print("\n" + "="*60)
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n🎯 Interactive Mode")
        print("Enter 'quit' or 'exit' to stop, 'help' for assistance")
        print("-" * 50)
        
        while True:
            try:
                # Get question
                print("\n📝 Enter the TOEFL Speaking question:")
                question = input("> ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif question.lower() == 'help':
                    self.show_help()
                    continue
                elif not question:
                    print("⚠️  Please enter a question.")
                    continue
                
                # Get student response
                print("\n🗣️  Enter the student's response:")
                answer = input("> ").strip()
                
                if not answer:
                    print("⚠️  Please enter a response.")
                    continue
                
                # Evaluate
                result = self.evaluate(question, answer)
                self.print_result(question, answer, result)
                
                # Ask if they want to continue
                print("\n🔄 Evaluate another response? (y/n)")
                continue_choice = input("> ").strip().lower()
                if continue_choice in ['n', 'no']:
                    print("👋 Goodbye!")
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show help information"""
        print("\n📚 HELP - TOEFL Independent Speaking Judge")
        print("="*50)
        print("This tool evaluates TOEFL Independent Speaking responses on a 0-4 scale.")
        print("\nScoring Rubric:")
        print("🌟 Score 4: Excellent - Sustained, coherent discourse")
        print("✅ Score 3: Good - Mostly coherent with minor limitations")  
        print("⚠️  Score 2: Limited - Basic ideas with limited development")
        print("❌ Score 1: Very Limited - Major language/content issues")
        print("💀 Score 0: No attempt or unrelated to topic")
        print("\nTips:")
        print("• Enter the exact question as given in TOEFL")
        print("• Provide the student's transcribed spoken response")
        print("• The model evaluates based on language use, delivery, and topic development")
        print("• Type 'quit' or 'exit' to stop")
        print("="*50)

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='TOEFL Independent Speaking Judge')
    parser.add_argument('--model', default='mlx-community/Llama-3.2-3B-Instruct-4bit', 
                       help='Model path')
    parser.add_argument('--adapter', default='toefl_judge_adapter', 
                       help='Adapter path')
    parser.add_argument('--question', '-q', help='TOEFL question')
    parser.add_argument('--answer', '-a', help='Student response')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize judge
    judge = SimpleTOEFLJudge(model_path=args.model, adapter_path=args.adapter)
    
    # Command line mode
    if args.question and args.answer:
        result = judge.evaluate(args.question, args.answer)
        judge.print_result(args.question, args.answer, result)
    
    # Interactive mode (default)
    else:
        judge.interactive_mode()

# Sample questions for testing
SAMPLE_QUESTIONS = [
    "Do you agree or disagree with the following statement: it is better to work in teams than to work alone? Use specific examples to support your answer.",
    "Some people prefer to live in small towns, while others prefer big cities. Which do you prefer and why?",
    "Do you think students should be required to wear uniforms in school? Explain your opinion with specific reasons.",
    "Would you rather have a job that pays well but is stressful, or a job that pays less but is enjoyable? Explain your choice.",
    "Some people believe that technology has made our lives easier, while others think it has made life more complicated. What is your opinion?"
]

if __name__ == "__main__":
    print("🚀 Quick Test Mode")
    print("Want to try a sample question? Here are some examples:")
    for i, q in enumerate(SAMPLE_QUESTIONS[:3], 1):
        print(f"{i}. {q}")
    print("\nStarting TOEFL Judge...")
    print("-" * 70)
    
    main()