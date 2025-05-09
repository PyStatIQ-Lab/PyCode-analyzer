import streamlit as st
import requests
import hashlib
import json
import time
from typing import Dict, Any
from functools import lru_cache

# ========== CONFIGURATION ==========
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Free Hugging Face model
CACHE_SIZE = 100
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
DEFAULT_TOKEN = "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"  # Optional: Set a default token for testing

# ========== HELPER FUNCTIONS ==========
def get_score_class(score: int) -> str:
    """Return CSS class based on score value"""
    if score >= 80: return "excellent"
    elif score >= 60: return "good"
    elif score >= 40: return "average"
    else: return "poor"

def get_risk_profile_type(risk_score: int) -> Dict[str, Any]:
    """Determine trader risk profile based on score"""
    if risk_score >= 80:
        return {"type": "Ultra-Conservative", "description": "Extremely risk-averse", "icon": "🛡️", "color": "blue"}
    elif risk_score >= 60:
        return {"type": "Conservative", "description": "Prefers low-risk investments", "icon": "☂️", "color": "green"}
    elif risk_score >= 40:
        return {"type": "Moderate", "description": "Balances risk and return", "icon": "⚖️", "color": "orange"}
    else:
        return {"type": "Aggressive", "description": "Seeks high returns", "icon": "⚡", "color": "red"}

def generate_prompt(code: str) -> str:
    """Generate structured prompt for analysis"""
    return f"""<<SYS>>You are a trading algorithm analyst. Provide JSON analysis of this code:<</SYS>>

[INST]Analyze this trading code and return JSON with:
1. Ratings (1-100) for: data_accuracy, model_efficiency, problem_solving, logical_structure, risk_profile
2. List of pros and cons
3. Risk classification with justification

Format EXACTLY like this:
{{
  "description": "summary",
  "ratings": {{"data_accuracy": 75, "model_efficiency": 80, ...}},
  "pros": ["point1", "point2"],
  "cons": ["issue1", "issue2"],
  "risk_profile_classification": {{
    "type": "Moderate",
    "justification": "explanation"
  }}
}}

Code:
{code}[/INST]"""

def normalize_scores(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure scores are valid integers between 0-100"""
    if 'ratings' not in analysis:
        return analysis
        
    for metric in analysis['ratings']:
        try:
            score = int(analysis['ratings'][metric])
            analysis['ratings'][metric] = max(0, min(100, score))
        except (ValueError, TypeError):
            analysis['ratings'][metric] = 50  # Default score
            
    return analysis

def calculate_overall_score(ratings: Dict[str, int]) -> float:
    """Calculate weighted overall score"""
    weights = {
        'data_accuracy': 0.25,
        'model_efficiency': 0.2,
        'problem_solving': 0.2,
        'logical_structure': 0.2,
        'risk_profile': 0.15
    }
    return round(sum(ratings[field] * weights[field] for field in weights), 1)

# ========== CORE ANALYSIS FUNCTION ==========
@lru_cache(maxsize=CACHE_SIZE)
def analyze_code(code: str, hf_token: str) -> Dict[str, Any]:
    """Analyze code using Hugging Face API with multiple fallback strategies"""
    headers = {"Authorization": f"Bearer {hf_token}"}
    prompt = generate_prompt(code)
    
    # Try with different parameters if first attempt fails
    for attempt in range(3):
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1000,
                    "temperature": 0.3 if attempt < 2 else 0.1,  # More deterministic on last try
                    "return_full_text": False
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            # Handle rate limiting and model loading
            if response.status_code == 503:
                est_time = int(response.headers.get('estimated_time', 30))
                st.warning(f"Model is loading, waiting {est_time} seconds...")
                time.sleep(est_time)
                continue
                
            if response.status_code != 200:
                error_msg = response.json().get('error', response.text)
                st.error(f"API Error (Attempt {attempt + 1}): {error_msg}")
                time.sleep(2)
                continue
                
            # Try multiple strategies to extract JSON
            response_text = response.json()[0]['generated_text']
            
            # Strategy 1: Direct JSON parse
            try:
                analysis = json.loads(response_text)
                if all(field in analysis for field in ['description', 'ratings']):
                    return normalize_scores(analysis)
            except json.JSONDecodeError:
                pass
                
            # Strategy 2: Extract JSON from markdown code block
            if '```json' in response_text:
                json_str = response_text.split('```json')[1].split('```')[0]
                try:
                    analysis = json.loads(json_str)
                    if all(field in analysis for field in ['description', 'ratings']):
                        return normalize_scores(analysis)
                except (json.JSONDecodeError, IndexError):
                    pass
                    
            # Strategy 3: Find first/last braces
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                try:
                    analysis = json.loads(response_text[json_start:json_end])
                    if all(field in analysis for field in ['description', 'ratings']):
                        return normalize_scores(analysis)
                except json.JSONDecodeError:
                    pass
                    
            # If all strategies fail, show debugging info
            st.error(f"Could not extract valid JSON from response (Attempt {attempt + 1})")
            st.code(f"Raw response:\n{response_text}")
            
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(2)
            continue
            
    return None

# ========== STREAMLIT UI ==========
def main():
    st.set_page_config(
        page_title="Trading Code Analyzer",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Trading Algorithm Analyzer")
    st.caption("Analyze Python trading code using AI")

    # Token input
    with st.expander("🔑 API Settings", expanded=True):
        hf_token = st.text_input(
            "Hugging Face Token",
            type="password",
            value=DEFAULT_TOKEN,
            help="Get your token from https://huggingface.co/settings/tokens"
        )

    # Code input
    code = st.text_area(
        "Paste your Python trading code:",
        height=300,
        placeholder="""def moving_average_strategy(prices, window=50):
    signals = []
    for i in range(len(prices)):
        if i >= window:
            avg = sum(prices[i-window:i])/window
            signals.append('BUY' if prices[i] > avg else 'SELL')
    return signals"""
    )

    # Analysis button
    if st.button("🚀 Analyze Code", use_container_width=True):
        if not hf_token:
            st.error("Please enter your Hugging Face token")
        elif not code.strip():
            st.error("Please enter some code to analyze")
        else:
            with st.spinner("🔍 Analyzing code (may take 20-40 seconds)..."):
                analysis = analyze_code(code.strip(), hf_token)
                
            if analysis:
                display_results(analysis)
            else:
                st.error("Analysis failed. Please check your token and try again.")

def display_results(analysis: Dict[str, Any]):
    """Display analysis results beautifully"""
    overall_score = calculate_overall_score(analysis['ratings'])
    risk_profile = get_risk_profile_type(analysis['ratings']['risk_profile'])

    # Score header
    st.subheader("📈 Analysis Results")
    score_color = ("#4CAF50" if overall_score >= 80 else
                  "#8BC34A" if overall_score >= 60 else
                  "#FFC107" if overall_score >= 40 else "#F44336")
    
    st.markdown(f"""
    <div style="background:{score_color};color:white;padding:1rem;border-radius:10px;text-align:center;">
        <h2>Overall Score</h2>
        <h1>{overall_score}/100</h1>
    </div>
    """, unsafe_allow_html=True)

    # Main columns
    col1, col2 = st.columns(2)

    with col1:
        # Description
        st.markdown("### 📝 Algorithm Overview")
        st.write(analysis['description'])

        # Risk Profile
        st.markdown(f"### {risk_profile['icon']} Risk Profile: {analysis['risk_profile_classification']['type']}")
        st.write(risk_profile['description'])
        st.markdown(f"**Justification:** {analysis['risk_profile_classification']['justification']}")

    with col2:
        # Metrics
        st.markdown("### ⚖️ Quality Metrics")
        for metric, score in analysis['ratings'].items():
            st.markdown(f"**{metric.replace('_', ' ').title()}**")
            st.progress(score/100, text=f"{score}/100")

    # Pros/Cons
    st.markdown("### ✅ Strengths")
    for pro in analysis['pros']:
        st.success(f"• {pro}")

    st.markdown("### ⚠️ Weaknesses")
    for con in analysis['cons']:
        st.error(f"• {con}")

if __name__ == "__main__":
    main()
