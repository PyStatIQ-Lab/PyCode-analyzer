import streamlit as st
import subprocess
import json
import hashlib
from typing import Dict, Any

# Configuration
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "mistral"  # or "ollama3"

# Initialize session state
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "overall_score" not in st.session_state:
    st.session_state.overall_score = None

# Helper Functions
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
            "icon": "🛡️",
            "color": "blue"
        }
    elif risk_score >= 60:
        return {
            "type": "Conservative",
            "description": "Prefers low-risk investments with steady returns",
            "icon": "☂️",
            "color": "green"
        }
    elif risk_score >= 40:
        return {
            "type": "Moderate",
            "description": "Balances risk and return, accepts some volatility",
            "icon": "⚖️",
            "color": "orange"
        }
    else:
        return {
            "type": "Aggressive",
            "description": "Seeks high returns and accepts significant risk",
            "icon": "⚡",
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

@st.cache_data(ttl=3600)
def analyze_code_with_ollama(code: str) -> Dict[str, Any]:
    """Send code to Ollama for comprehensive analysis with caching"""
    prompt = generate_prompt(code)
    code_hash = hashlib.md5(code.encode('utf-8')).hexdigest()
    
    try:
        curl_command = [
            "curl",
            "-s",
            "-X", "POST",
            f"{OLLAMA_HOST}/api/generate",
            "-d", json.dumps({
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.3,  # More deterministic output
                    "top_p": 0.9,
                    "seed": int(code_hash[:8], 16) % 1000000  # Consistent seed per code
                }
            }),
            "-H", "Content-Type: application/json"
        ]
        
        result = subprocess.run(curl_command, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise Exception(f"Curl command failed: {result.stderr}")
            
        response = json.loads(result.stdout)
        
        try:
            if "response" in response:
                analysis = json.loads(response["response"])
            else:
                analysis = response
                
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
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            st.error(f"Error parsing response: {e}")
            st.error(f"Raw response: {response}")
            return None
            
    except Exception as e:
        st.error(f"Error communicating with Ollama: {e}")
        return None

# Streamlit UI
st.set_page_config(
    page_title="Trading Code Risk Analyzer",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📊 Trading Code Risk Analyzer")
st.markdown("Comprehensive analysis of trading algorithms with risk profile classification")

# Code Input
code = st.text_area(
    "Enter Trading Algorithm Code:",
    height=300,
    placeholder="Paste your Python trading code here..."
)

if st.button("Analyze Code", type="primary"):
    if not code.strip():
        st.error("Please enter Python code to analyze")
    else:
        with st.spinner("Analyzing code..."):
            analysis = analyze_code_with_ollama(code)
            
            if not analysis:
                st.error("Failed to analyze code. Please ensure Ollama is running and try again.")
            else:
                st.session_state.analysis = analysis
                st.session_state.overall_score = calculate_overall_score(analysis['ratings'])
                st.success("Analysis completed!")

# Display Results
if st.session_state.analysis:
    st.divider()
    st.header("Analysis Results")
    
    # Overall Score
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📈 Quality Metrics")
    with col2:
        score_class = get_score_class(st.session_state.overall_score)
        st.markdown(f"""
        <div style="
            width: 100px;
            height: 100px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: {'#d1fae5' if score_class == 'excellent' else 
                             '#dbeafe' if score_class == 'good' else 
                             '#fef3c7' if score_class == 'average' else '#fee2e2'};
            color: {'#065f46' if score_class == 'excellent' else 
                   '#1e40af' if score_class == 'good' else 
                   '#92400e' if score_class == 'average' else '#991b1b'};
            margin: 0 auto;
        ">
            <div style="font-size: 2.2rem; line-height: 1;">{st.session_state.overall_score}</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">Overall Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Results Grid
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Description", 
        "👤 Risk Profile", 
        "⭐ Ratings", 
        "✅ Strengths", 
        "⚠️ Weaknesses"
    ])
    
    with tab1:
        st.write(st.session_state.analysis['description'])
    
    with tab2:
        risk_profile = get_risk_profile_type(st.session_state.analysis['ratings']['risk_profile'])
        risk_class = risk_profile['type'].lower().replace(' ', '-')
        
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            border-radius: 8px;
            background-color: #f8f9fa;
        ">
            <div style="
                font-size: 2rem;
                width: 60px;
                height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                background-color: {'#e6f3ff' if risk_class == 'ultra-conservative' else 
                                 '#e6f7ed' if risk_class == 'conservative' else 
                                 '#fff3e0' if risk_class == 'moderate' else '#fce8e6'};
                color: {'#1a73e8' if risk_class == 'ultra-conservative' else 
                       '#34a853' if risk_class == 'conservative' else 
                       '#f9ab00' if risk_class == 'moderate' else '#d93025'};
            ">
                {risk_profile['icon']}
            </div>
            <div style="flex: 1;">
                <h4 style="margin-bottom: 0.5rem; font-size: 1.2rem;">{risk_profile['type']}</h4>
                <p>{risk_profile['description']}</p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #eee;">
                    <h5 style="margin-bottom: 0.5rem; font-size: 1rem;">Justification:</h5>
                    <p>{st.session_state.analysis['risk_profile_classification']['justification']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        for metric, score in st.session_state.analysis['ratings'].items():
            score_class = get_score_class(score)
            st.markdown(f"""
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            ">
                <div style="flex: 1;">
                    <span style="font-weight: 500; display: block; margin-bottom: 0.3rem;">
                        {metric.replace('_', ' ').title()}
                    </span>
                    <div style="
                        height: 8px;
                        background-color: #e9ecef;
                        border-radius: 4px;
                        overflow: hidden;
                    ">
                        <div style="
                            height: 100%;
                            width: {score}%;
                            background-color: #4361ee;
                            border-radius: 4px;
                        "></div>
                    </div>
                </div>
                <span style="
                    font-weight: bold;
                    margin-left: 1rem;
                    min-width: 40px;
                    text-align: right;
                    color: {'#065f46' if score_class == 'excellent' else 
                           '#1e40af' if score_class == 'good' else 
                           '#92400e' if score_class == 'average' else '#991b1b'};
                ">
                    {score}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        for pro in st.session_state.analysis['pros']:
            st.success(f"✅ {pro}")
    
    with tab5:
        for con in st.session_state.analysis['cons']:
            st.error(f"⚠️ {con}")

# Footer
st.divider()
st.markdown("""
<p style="text-align: center; color: #666;">
    Trading Code Risk Analyzer - Powered by Ollama and Mistral
</p>
""", unsafe_allow_html=True)
