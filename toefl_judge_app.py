import streamlit as st
from mlx_lm import load, generate
import re
import json
from datetime import datetime
import pandas as pd

@st.cache_resource
def load_toefl_model(model_path, adapter_path):
    """Load the fine-tuned TOEFL judge model - cached globally"""
    try:
        model, tokenizer = load(
            path_or_hf_repo=model_path,
            adapter_path=adapter_path
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

class TOEFLJudgeApp:
    def __init__(self, model_path, adapter_path):
        self.model_path = model_path
        self.adapter_path = adapter_path
        # Load model using cached function
        self.model, self.tokenizer = load_toefl_model(model_path, adapter_path)
    
    def create_evaluation_prompt(self, question, answer):
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
    
    def extract_score_and_feedback(self, response):
        """Extract score and feedback from model response"""
        # Extract score
        score_match = re.search(r'Score:\s*(\d+)', response)
        score = int(score_match.group(1)) if score_match else None
        
        # Extract explanation (everything after "Score: X")
        if score_match:
            explanation = response[score_match.end():].strip()
        else:
            explanation = response
        
        return score, explanation
    
    def evaluate_response(self, question, answer):
        """Evaluate a single TOEFL response"""
        if not self.model or not self.tokenizer:
            return None, "Model not loaded"
        
        try:
            prompt = self.create_evaluation_prompt(question, answer)
            
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=400
            )
            
            score, explanation = self.extract_score_and_feedback(response)
            
            return {
                'score': score,
                'explanation': explanation,
                'full_response': response,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return None, f"Error during evaluation: {e}"
    
    def save_evaluation(self, question, answer, result):
        """Save evaluation to CSV for record keeping"""
        evaluation_data = {
            'timestamp': result['timestamp'],
            'question': question,
            'answer': answer,
            'score': result['score'],
            'explanation': result['explanation']
        }
        
        # Append to CSV file
        df = pd.DataFrame([evaluation_data])
        try:
            existing_df = pd.read_csv('toefl_evaluations.csv')
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass  # First entry
        
        df.to_csv('toefl_evaluations.csv', index=False)

def main():
    st.set_page_config(
        page_title="TOEFL Independent Speaking Judge",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 TOEFL Independent Speaking Judge")
    st.markdown("### Automated scoring based on official TOEFL iBT rubric")
    
    # Initialize the app
    MODEL_PATH = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    ADAPTER_PATH = "toefl_judge_adapter"
    
    if 'toefl_app' not in st.session_state:
        with st.spinner("Loading TOEFL Judge model..."):
            st.session_state.toefl_app = TOEFLJudgeApp(MODEL_PATH, ADAPTER_PATH)
    
    app = st.session_state.toefl_app
    
    # Check if model loaded successfully
    if not app.model or not app.tokenizer:
        st.error("❌ Failed to load the TOEFL model. Please check your model and adapter paths.")
        st.stop()
    
    # Sidebar with model performance stats
    with st.sidebar:
        st.header("📊 Model Performance")
        st.metric("Exact Accuracy", "86.1%")
        st.metric("Adjacent Accuracy", "99.7%")
        st.metric("Correlation", "0.943")
        st.metric("Mean Error", "0.142 points")
        
        st.markdown("---")
        st.header("📋 Scoring Rubric")
        st.markdown("""
        **Score 4:** Excellent response with sustained, coherent discourse
        
        **Score 3:** Good response, mostly coherent with minor limitations
        
        **Score 2:** Basic response with limited development
        
        **Score 1:** Very limited response with major issues
        
        **Score 0:** No attempt or unrelated to topic
        """)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Question input
        question = st.text_area(
            "TOEFL Speaking Question:",
            placeholder="Enter the Independent Speaking question here...",
            height=100,
            value=""
        )
        
        # Student response input
        answer = st.text_area(
            "Student Response:",
            placeholder="Enter the student's spoken response (transcribed) here...",
            height=200,
            value=""
        )
        
        # Evaluation button
        if st.button("🎯 Evaluate Response", type="primary"):
            if question.strip() and answer.strip():
                with st.spinner("Evaluating response..."):
                    result = app.evaluate_response(question.strip(), answer.strip())
                
                if result and isinstance(result, dict) and result.get('score') is not None:
                    st.session_state.last_result = result
                    st.session_state.last_question = question.strip()
                    st.session_state.last_answer = answer.strip()
                    
                    # Save evaluation
                    app.save_evaluation(question.strip(), answer.strip(), result)
                    
                    st.success("✅ Evaluation complete!")
                    st.rerun()  # Refresh to show results
                else:
                    error_msg = result[1] if isinstance(result, tuple) else "Unknown error"
                    st.error(f"❌ Evaluation failed: {error_msg}")
            else:
                st.warning("⚠️ Please enter both question and response.")
    
    with col2:
        st.header("📊 Results")
        
        if 'last_result' in st.session_state:
            result = st.session_state.last_result
            
            # Display score with color coding
            score = result['score']
            if score == 4:
                st.success(f"## Score: {score}/4 🌟")
            elif score == 3:
                st.info(f"## Score: {score}/4 ✅")
            elif score == 2:
                st.warning(f"## Score: {score}/4 ⚠️")
            else:
                st.error(f"## Score: {score}/4 ❌")
            
            # Display explanation
            st.markdown("### 📋 Detailed Feedback")
            st.markdown(result['explanation'])
            
            # Show question and answer that were evaluated
            with st.expander("📝 View Evaluated Content"):
                st.write(f"**Question:** {st.session_state.get('last_question', 'N/A')}")
                st.write(f"**Response:** {st.session_state.get('last_answer', 'N/A')}")
            
        else:
            st.info("👆 Enter a question and response above to get started")
    
    # Recent evaluations section
    st.markdown("---")
    st.header("📈 Recent Evaluations")
    
    try:
        recent_df = pd.read_csv('toefl_evaluations.csv')
        if not recent_df.empty:
            # Show last 5 evaluations
            recent_df = recent_df.tail(5).sort_values('timestamp', ascending=False)
            
            for _, row in recent_df.iterrows():
                with st.expander(f"Score {row['score']}/4 - {str(row['timestamp'])[:19]}"):
                    st.write(f"**Question:** {str(row['question'])[:100]}...")
                    st.write(f"**Response:** {str(row['answer'])[:150]}...")
                    st.write(f"**Score:** {row['score']}/4")
                    st.write(f"**Feedback:** {str(row['explanation'])[:200]}...")
        else:
            st.info("No evaluations recorded yet.")
    except FileNotFoundError:
        st.info("No evaluation history available.")
    except Exception as e:
        st.warning(f"Could not load evaluation history: {e}")

if __name__ == "__main__":
    main()

# To run this app:
# streamlit run toefl_judge_app.py