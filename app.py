import streamlit as st
import requests
import hashlib
import json
from typing import Dict, Any
from functools import lru_cache

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Free model available on Hugging Face
CACHE_SIZE = 100  # Number of analyses to cache
HF_API_TOKEN = "your_huggingface_token"  # Get from https://huggingface.co/settings/tokens
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

def get_score_class(score: int) -> str:
    """Return CSS class based on score value"""
    if score >= 80:
        return "excellent"
    elif score >= 60:
        return "good"
    elif score >= 40:
        return "average"
    else:
        return "poor"

def get_risk_profile_type(risk_score: int) -> Dict[str, Any]:
    """Determine the trader risk profile based on risk score"""
    if risk_score >= 80:
        return {
            "type": "Ultra-Conservative",
            "description": "Extremely risk-averse, prioritizes capital preservation above all else",
            "icon": "shield",
            "color": "blue"
        }
    elif risk_score >= 60:
        return {
            "type": "Conservative",
            "description": "Prefers low-risk investments with steady returns",
            "icon": "umbrella",
            "color": "green"
        }
    elif risk_score >= 40:
        return {
            "type": "Moderate",
            "description": "Balances risk and return, accepts some volatility",
            "icon": "scales",
            "color": "orange"
        }
    else:
        return {
            "type": "Aggressive",
            "description": "Seeks high returns and accepts significant risk",
            "icon": "bolt",
            "color": "red"
        }

def generate_prompt(code: str) -> str:
    """Generate a structured prompt for consistent analysis"""
    criteria = {
        "data_accuracy": "Completeness of financial data, handling of missing data, data validation",
        "model_efficiency": "Computational complexity, memory usage, optimization techniques",
        "problem_solving": "Edge case handling, error recovery, robustness to market changes",
        "logical_structure": "Code organization, modularity, readability, documentation",
        "risk_profile": "Risk management features, position sizing, stop-loss mechanisms"
    }
    
    prompt = f"""<<SYS>>You are a trading algorithm analysis assistant. Analyze the following Python trading code systematically and provide consistent ratings:<</SYS>>

[INST]
1. Evaluate these aspects STRICTLY on a scale of 1-100:
- Data Accuracy: {criteria['data_accuracy']}
- Model Efficiency: {criteria['model_efficiency']}
- Problem Solving: {criteria['problem_solving']}
- Logical Structure: {criteria['logical_structure']}
- Risk Profile: {criteria['risk_profile']}

2. Use these evaluation guidelines:
- Start with 50 as a neutral score
- Add/subtract points based on specific features
- Deduct points for each identified issue
- Cap scores at 100

3. Provide EXPLICIT justification for each score

Format your response as EXACTLY this JSON structure:
{{
  "description": "...",
  "ratings": {{
    "data_accuracy": int (1-100),
    "model_efficiency": int (1-100),
    "problem_solving": int (1-100),
    "logical_structure": int (1-100),
    "risk_profile": int (1-100)
  }},
  "pros": ["...", "...", "..."],
  "cons": ["...", "...", "..."],
  "risk_profile_classification": {{
    "type": "...",
    "justification": "..."
  }}
}}

Code to analyze:
{code}
[/INST]"""
    return prompt

def normalize_scores(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure scores fall within reasonable ranges and are integers"""
    if 'ratings' not in analysis:
        return analysis
        
    for metric in analysis['ratings']:
        try:
            score = int(analysis['ratings'][metric])
            analysis['ratings'][metric] = max(0, min(100, score))
        except (ValueError, TypeError):
            analysis['ratings'][metric] = 50  # Default neutral score
            
    return analysis

def calculate_overall_score(ratings: Dict[str, int]) -> float:
    """Calculate weighted overall score out of 100"""
    weights = {
        'data_accuracy': 0.25,
        'model_efficiency': 0.2,
        'problem_solving': 0.2,
        'logical_structure': 0.2,
        'risk_profile': 0.15
    }
    
    overall = sum(ratings[field] * weights[field] for field in weights)
    return round(overall, 1)

def get_code_hash(code: str) -> str:
    """Generate a consistent hash for caching"""
    return hashlib.md5(code.encode('utf-8')).hexdigest()

@lru_cache(maxsize=CACHE_SIZE)
def analyze_code_with_hf(code: str) -> Dict[str, Any]:
    """Send code to Hugging Face Inference API for analysis"""
    prompt = generate_prompt(code)
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 1000, "temperature": 0.3}}
        )
        
        if response.status_code != 200:
            st.error(f"API Error: {response.text}")
            return None
            
        analysis_text = response.json()[0]['generated_text']
        
        # Extract just the JSON part from the response
        json_start = analysis_text.find('{')
        json_end = analysis_text.rfind('}') + 1
        json_str = analysis_text[json_start:json_end]
        
        analysis = json.loads(json_str)
        
        # Validate and normalize the response
        required = ['description', 'pros', 'cons', 'ratings', 'risk_profile_classification']
        for field in required:
            if field not in analysis:
                raise ValueError(f"Missing field {field} in response")
                
        analysis = normalize_scores(analysis)
        
        # Ensure risk profile classification exists
        if 'type' not in analysis['risk_profile_classification']:
            analysis['risk_profile_classification']['type'] = "Moderate"
        if 'justification' not in analysis['risk_profile_classification']:
            analysis['risk_profile_classification']['justification'] = "No specific justification provided"
            
        return analysis
        
    except Exception as e:
        st.error(f"Error analyzing code: {str(e)}")
        return None

def display_analysis_results(analysis: Dict[str, Any], overall_score: float):
    """Display the analysis results in Streamlit"""
    score_class = get_score_class(overall_score)
    risk_type = get_risk_profile_type(analysis['ratings']['risk_profile'])
    
    # Overall score
    st.markdown(f"""
    <div style="
        background: {'#4CAF50' if score_class == 'excellent' else 
                   '#8BC34A' if score_class == 'good' else 
                   '#FFC107' if score_class == 'average' else '#F44336'};
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    ">
        <h2>Overall Score</h2>
        <h1>{overall_score}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for the results
    col1, col2 = st.columns(2)
    
    with col1:
        # Algorithm Description
        st.subheader("📝 Algorithm Description")
        st.write(analysis['description'])
        
        # Risk Profile
        st.subheader(f"👤 Trader Risk Profile: {analysis['risk_profile_classification']['type']}")
        st.write(risk_type['description'])
        st.write(f"**Justification:** {analysis['risk_profile_classification']['justification']}")
        
    with col2:
        # Quality Metrics
        st.subheader("⭐ Quality Metrics")
        for metric, score in analysis['ratings'].items():
            st.write(f"**{metric.replace('_', ' ').title()}**")
            st.progress(score / 100)
            st.caption(f"Score: {score}")
        
    # Pros and Cons
    st.subheader("✅ Strengths")
    for pro in analysis['pros']:
        st.success(f"• {pro}")
    
    st.subheader("⚠️ Risk Factors")
    for con in analysis['cons']:
        st.error(f"• {con}")

def main():
    st.set_page_config(
        page_title="Python Trading Code Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Trading Code Risk Analyzer")
    st.markdown("Comprehensive analysis of trading algorithms with risk profile classification")
    
    # Add link to get Hugging Face token
    st.markdown("""
    <small>You need a Hugging Face API token. Get one at 
    <a href="https://huggingface.co/settings/tokens" target="_blank">https://huggingface.co/settings/tokens</a>
    </small>
    """, unsafe_allow_html=True)
    
    # Let users input their own token
    hf_token = st.text_input("Enter your Hugging Face API token:", type="password")
    
    code = st.text_area(
        "Enter Trading Algorithm Code:",
        height=300,
        placeholder="Paste your Python trading code here..."
    )
    
    if st.button("Analyze Code"):
        if not hf_token:
            st.error("Please enter your Hugging Face API token")
        elif not code.strip():
            st.error("Please enter Python code to analyze")
        else:
            with st.spinner("Analyzing code (this may take 20-30 seconds)..."):
                # Set the token globally
                global HF_API_TOKEN
                HF_API_TOKEN = hf_token
                
                analysis = analyze_code_with_hf(code.strip())
                
                if not analysis:
                    st.error("Failed to analyze code. Please check your token and try again.")
                else:
                    overall_score = calculate_overall_score(analysis['ratings'])
                    st.success("Analysis complete!")
                    display_analysis_results(analysis, overall_score)

if __name__ == "__main__":
    main()
