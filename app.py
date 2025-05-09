import streamlit as st
import ollama
import hashlib
import json
from typing import Dict, Any
from functools import lru_cache

# Configuration
MODEL_NAME = "mistral"  # or "ollama3"
CACHE_SIZE = 100  # Number of analyses to cache

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
    
    prompt = f"""
Analyze the following Python trading code systematically and provide consistent ratings:

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
"""
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
def analyze_code_with_ollama(code: str) -> Dict[str, Any]:
    """Send code to Ollama for comprehensive analysis with caching"""
    prompt = generate_prompt(code)
    code_hash = get_code_hash(code)
    
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            format="json",
            options={
                "temperature": 0.3,
                "top_p": 0.9,
                "seed": int(code_hash[:8], 16) % 1000000
            }
        )
        
        analysis = json.loads(response["response"])
        
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
        st.error(f"Error analyzing code: {e}")
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
    
    code = st.text_area(
        "Enter Trading Algorithm Code:",
        height=300,
        placeholder="Paste your Python trading code here..."
    )
    
    if st.button("Analyze Code"):
        if not code.strip():
            st.error("Please enter Python code to analyze")
        else:
            with st.spinner("Analyzing code..."):
                analysis = analyze_code_with_ollama(code.strip())
                
                if not analysis:
                    st.error("Failed to analyze code. Please ensure Ollama is running and try again.")
                else:
                    overall_score = calculate_overall_score(analysis['ratings'])
                    st.success("Analysis complete!")
                    display_analysis_results(analysis, overall_score)

if __name__ == "__main__":
    main()
