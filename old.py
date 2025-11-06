'''

import streamlit as st
import pickle
import re
import string
from datetime import datetime
import numpy as np
import requests
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Get your API key
env_api_key = os.getenv("FACTCHECK_API_KEY")


# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)


# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-top: 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .fact-check-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
        background-color: #f0f8ff;
    }
    .claim-item {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background-color: white;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Google Fact Check API Integration
# ----------------------------
class FactCheckAPI:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    def search_claims(self, query: str, language: str = "en") -> List[Dict]:
        """Search for fact-checked claims related to the query"""
        if not self.api_key:
            return []
        
        try:
            params = {
                'query': query[:200],  # Limit query length
                'languageCode': language,
                'key': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('claims', [])
            else:
                st.warning(f"Fact Check API returned status code: {response.status_code}")
                return []
        except requests.exceptions.Timeout:
            st.warning("⏱️ Fact check API request timed out")
            return []
        except Exception as e:
            st.warning(f"⚠️ Fact check API error: {str(e)}")
            return []
    
    def extract_key_phrases(self, text: str, max_phrases: int = 3) -> List[str]:
        """Extract key phrases from text for fact checking"""
        # Remove common words and extract meaningful phrases
        text = text.lower()
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Extract potential claims (sentences with specific keywords)
        claim_keywords = ['announced', 'revealed', 'discovered', 'confirmed', 
                         'reported', 'said', 'stated', 'claimed', 'found']
        
        potential_claims = []
        for sentence in sentences[:10]:  # Check first 10 sentences
            if any(keyword in sentence for keyword in claim_keywords):
                # Clean and truncate
                claim = ' '.join(sentence.split()[:15])  # First 15 words
                if len(claim) > 20:  # Minimum length
                    potential_claims.append(claim)
        
        return potential_claims[:max_phrases]
    
    def format_fact_check_results(self, claims: List[Dict]) -> List[Dict]:
        """Format fact check results for display"""
        formatted = []
        
        for claim in claims[:5]:  # Limit to 5 results
            try:
                claim_text = claim.get('text', 'N/A')
                claimant = claim.get('claimant', 'Unknown')
                claim_date = claim.get('claimDate', 'Unknown date')
                
                # Get claim reviews
                reviews = claim.get('claimReview', [])
                
                for review in reviews[:2]:  # Limit to 2 reviews per claim
                    publisher = review.get('publisher', {}).get('name', 'Unknown')
                    rating = review.get('textualRating', 'Not rated')
                    title = review.get('title', 'No title')
                    url = review.get('url', '#')
                    
                    formatted.append({
                        'claim': claim_text,
                        'claimant': claimant,
                        'date': claim_date,
                        'publisher': publisher,
                        'rating': rating,
                        'title': title,
                        'url': url
                    })
            except Exception as e:
                continue
        
        return formatted

# ----------------------------
# Load Model and Vectorizer
# ----------------------------
@st.cache_resource
def load_model():
    try:
        with open("fake_news_model_v2.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer_v2.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run `train_model.py` first.")
        st.info("""
        **How to train the model:**
        1. Download the dataset from Kaggle: 
           https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
        2. Place Fake.csv and True.csv in the same directory
        3. Run: `python train_model.py`
        4. Then run: `streamlit run app.py`
        """)
        st.stop()

model, vectorizer = load_model()

# ----------------------------
# Text Preprocessing
# ----------------------------
def preprocess_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    return text

# ----------------------------
# Check for Trusted Sources
# ----------------------------
def check_trusted_source(text):
    """Check if text mentions trusted news sources"""
    trusted_sources = [
        "bbc", "reuters", "associated press", "apnews", "nasa", "nytimes", 
        "new york times", "guardian", "cnn", "washington post", "washingtonpost",
        "forbes", "bloomberg", "wall street journal", "wsj", "npr", "pbs",
        "abc news", "nbc news", "cbs news", "the economist", "time magazine"
    ]
    text_lower = text.lower()
    return any(src in text_lower for src in trusted_sources)

# ----------------------------
# Analyze Risk Factors
# ----------------------------
def analyze_risk_factors(text):
    """Identify potential red flags in the text"""
    warnings = []
    
    exclamation_count = text.count("!")
    if exclamation_count > 3:
        warnings.append(f"⚠️ Excessive exclamation marks ({exclamation_count})")
    
    clickbait_words = ["shocking", "breaking", "urgent", "unbelievable", "you won't believe", 
                       "doctors hate", "this one trick", "what happens next"]
    found_clickbait = [word for word in clickbait_words if word in text.lower()]
    if found_clickbait:
        warnings.append(f"⚠️ Clickbait keywords: {', '.join(found_clickbait)}")
    
    viral_phrases = ["share this", "share now", "spread the word", "before it's deleted"]
    found_viral = [phrase for phrase in viral_phrases if phrase in text.lower()]
    if found_viral:
        warnings.append(f"⚠️ Viral call-to-action: {', '.join(found_viral)}")
    
    word_count = len(text.split())
    if word_count < 50:
        warnings.append(f"⚠️ Very short content ({word_count} words)")
    
    if any(word.isupper() and len(word) > 3 for word in text.split()):
        warnings.append("⚠️ Excessive capitalization detected")
    
    if not warnings:
        warnings.append("✅ No major red flags detected")
    
    return warnings

# ----------------------------
# Prediction Function
# ----------------------------
def predict_news(news_text, confidence_threshold=0.55):
    """Predict if news is fake or real"""
    clean_text = preprocess_text(news_text)
    
    if len(clean_text.strip()) == 0:
        return None, None, False, "uncertain"
    
    tfidf = vectorizer.transform([clean_text])
    proba = model.predict_proba(tfidf)[0]
    
    fake_prob = proba[0]
    real_prob = proba[1]
    
    trusted = check_trusted_source(news_text)
    
    if trusted:
        if real_prob >= 0.40:
            boost = 0.25
        else:
            boost = 0.20
        real_prob = min(real_prob + boost, 0.98)
        fake_prob = max(1.0 - real_prob, 0.02)
    
    authoritative_keywords = ['nasa', 'study', 'research', 'university', 'scientists', 
                              'according to', 'officials', 'data', 'announced']
    has_authority = sum(1 for kw in authoritative_keywords if kw in news_text.lower()) >= 2
    
    if has_authority and real_prob >= 0.35:
        real_prob = min(real_prob + 0.15, 0.98)
        fake_prob = max(1.0 - real_prob, 0.02)
    
    max_prob = max(fake_prob, real_prob)
    
    if max_prob < confidence_threshold:
        prediction = "uncertain"
        label = None
    elif real_prob > fake_prob:
        prediction = "real"
        label = 1
    else:
        prediction = "fake"
        label = 0
    
    return label, (fake_prob, real_prob), trusted, prediction

# ----------------------------
# Main UI
# ----------------------------
st.markdown("<h1 class='main-header'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-powered fake news detection with Google Fact Check verification</p>", unsafe_allow_html=True)

st.write("")

# API Key Configuration in Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Check if loaded from .env
    if env_api_key:
        st.success("✅ API Key loaded from .env")
        st.caption(f"Key: {env_api_key[:10]}...")
        api_key = None
    else:
        api_key = st.text_input(
            "Google Fact Check API Key (Optional)",
            type="password",
            help="Get your free API key from: https://console.cloud.google.com/"
        )
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.info("ℹ️ Add API key to enable fact checking")

# Initialize Fact Check API
effective_api_key = env_api_key if env_api_key else (api_key if api_key else None)
fact_checker = FactCheckAPI(effective_api_key)


# Create tabs for different input methods
tab1, tab2 = st.tabs(["📝 Full Article", "🔍 Quick Check"])

with tab1:
    st.write("### Enter Complete News Article")
    news_title = st.text_input("**News Title**", placeholder="Enter the headline or title")
    news_content = st.text_area("**News Content**", height=250, 
                                placeholder="Paste the full content here...")

with tab2:
    st.write("### Quick Text Analysis")
    quick_text = st.text_area("**Paste any news text**", height=200,
                             placeholder="Paste any news snippet...")

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze News", use_container_width=True)

if analyze_button:
    # Determine which input to use
    if news_title or news_content:
        full_text = f"{news_title} {news_content}".strip()
    else:
        full_text = quick_text.strip()
    
    if not full_text:
        st.warning("⚠️ Please enter some news content to analyze.")
    else:
        with st.spinner("🔄 Analyzing content..."):
            pred, proba, trusted, prediction_type = predict_news(full_text, confidence_threshold=0.60)
            
            if pred is None and prediction_type != "uncertain":
                st.error("❌ Unable to analyze the text. Please enter valid content.")
            else:
                fake_prob = proba[0] * 100
                real_prob = proba[1] * 100
                
                st.write("---")
                
                # Main Result
                st.write("## 📊 AI Model Analysis")
                
                if prediction_type == "uncertain":
                    st.warning(f"""
                    ### ⚠️ **UNCERTAIN PREDICTION**
                    The model cannot confidently classify this content.
                    
                    **Confidence levels:**
                    - Real: {real_prob:.1f}%
                    - Fake: {fake_prob:.1f}%
                    """)
                elif prediction_type == "real":
                    st.success(f"""
                    ### ✅ This appears to be **REAL NEWS**
                    **Confidence: {real_prob:.1f}%**
                    """)
                else:
                    st.error(f"""
                    ### 🚨 This appears to be **FAKE or MISLEADING**
                    **Confidence: {fake_prob:.1f}%**
                    """)
                
                if trusted:
                    st.info("ℹ️ **Trusted source detected** - confidence adjusted (+20-25%)")
                
                # Confidence metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fake Probability", f"{fake_prob:.1f}%")
                    st.progress(fake_prob / 100)
                with col2:
                    st.metric("Real Probability", f"{real_prob:.1f}%")
                    st.progress(real_prob / 100)
                
                # Google Fact Check Results
                st.write("---")
                st.write("## 🔍 Google Fact Check Verification")
                
                if effective_api_key:
                    with st.spinner("🌐 Checking with Google Fact Check API..."):
                        # Extract key phrases for fact checking
                        key_phrases = fact_checker.extract_key_phrases(full_text)
                        
                        if key_phrases:
                            st.info(f"🔎 Searching for fact checks on {len(key_phrases)} key claims...")
                            
                            all_results = []
                            for phrase in key_phrases:
                                claims = fact_checker.search_claims(phrase)
                                if claims:
                                    formatted = fact_checker.format_fact_check_results(claims)
                                    all_results.extend(formatted)
                            
                            if all_results:
                                st.success(f"✅ Found {len(all_results)} related fact checks!")
                                
                                for i, result in enumerate(all_results[:5], 1):
                                    with st.container():
                                        st.markdown(f"""
                                        <div class='claim-item'>
                                            <h4>Fact Check #{i}</h4>
                                            <p><strong>Claim:</strong> {result['claim'][:200]}...</p>
                                            <p><strong>Claimant:</strong> {result['claimant']} ({result['date']})</p>
                                            <p><strong>Rating:</strong> <span style='color: #1f77b4; font-weight: bold;'>{result['rating']}</span></p>
                                            <p><strong>Verified by:</strong> {result['publisher']}</p>
                                            <p><a href="{result['url']}" target="_blank">📄 Read full fact check →</a></p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info("ℹ️ No related fact checks found in Google's database. This doesn't mean the content is false - it may simply not have been fact-checked yet.")
                        else:
                            st.info("ℹ️ Could not extract specific claims for fact checking.")
                else:
                    st.warning("⚠️ Add Google Fact Check API key in the sidebar to enable verification")
                    st.info("The Fact Check API is completely FREE and provides additional verification from professional fact-checkers worldwide.")
                
                # Risk Analysis
                st.write("---")
                st.write("## 🔍 Content Analysis")
                risk_factors = analyze_risk_factors(full_text)
                for factor in risk_factors:
                    st.write(factor)
                
                # Content Statistics
                st.write("## 📈 Content Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Word Count", len(full_text.split()))
                with col2:
                    st.metric("Character Count", len(full_text))
                with col3:
                    st.metric("Exclamation Marks", full_text.count("!"))
                with col4:
                    st.metric("Question Marks", full_text.count("?"))
                
                st.write("---")
                st.caption(f"🕒 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Tips
                with st.expander("💡 Tips for Verifying News"):
                    st.write("""
                    **How to spot fake news:**
                    1. ✓ Check multiple trusted sources (BBC, Reuters, AP)
                    2. ✓ Verify author credentials and publication date
                    3. ✓ Look for emotional manipulation and clickbait
                    4. ✓ Use reverse image search for photos
                    5. ✓ Check fact-checking websites (Snopes, FactCheck.org)
                    6. ✓ Verify the URL is legitimate
                    7. ✓ Look for credible expert quotes
                    8. ✓ Check the "About Us" page
                    9. ✓ Be skeptical of outrageous claims
                    10. ✓ Use Google Fact Check Explorer
                    """)

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.title("ℹ️ About This App")
st.sidebar.info("""
**Enhanced Fake News Detection System**

Combines machine learning with Google Fact Check API for comprehensive verification.

**Features:**
- AI Model Analysis (98% accuracy)
- Google Fact Check Integration
- Trusted source verification
- Content risk analysis
- Confidence scoring

**Model Details:**
- Algorithm: Logistic Regression
- Training: 40,000+ articles
- TF-IDF features (uni-grams & bi-grams)
""")

st.sidebar.markdown("---")
st.sidebar.write("**Dataset:** [Kaggle Fake & Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)")
st.sidebar.write("👨‍💻 **Course:** Cloud Computing (BITE412L)")
st.sidebar.write("🔬 **Version:** 3.0 (Enhanced)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool provides automated predictions. 
    Always verify with multiple sources and use fact-checking websites.</p>
</div>
""", unsafe_allow_html=True) '''

'''import streamlit as st
import pickle
import re
import string
from datetime import datetime
import numpy as np
import requests
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Get your API key
env_api_key = os.getenv("FACTCHECK_API_KEY")

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-top: 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .fact-check-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
        background-color: #f0f8ff;
    }
    .claim-item {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background-color: white;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Google Fact Check API Integration
# ----------------------------
class FactCheckAPI:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    def search_claims(self, query: str, language: str = "en") -> List[Dict]:
        """Search for fact-checked claims related to the query"""
        if not self.api_key:
            return []
        
        try:
            params = {
                'query': query[:200],
                'languageCode': language,
                'key': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('claims', [])
            else:
                st.warning(f"Fact Check API returned status code: {response.status_code}")
                return []
        except requests.exceptions.Timeout:
            st.warning("⏱️ Fact check API request timed out")
            return []
        except Exception as e:
            st.warning(f"⚠️ Fact check API error: {str(e)}")
            return []
    
    def extract_key_phrases(self, text: str, max_phrases: int = 3) -> List[str]:
        """Extract key phrases from text for fact checking"""
        text = text.lower()
        sentences = re.split(r'[.!?]+', text)
        
        claim_keywords = ['announced', 'revealed', 'discovered', 'confirmed', 
                         'reported', 'said', 'stated', 'claimed', 'found']
        
        potential_claims = []
        for sentence in sentences[:10]:
            if any(keyword in sentence for keyword in claim_keywords):
                claim = ' '.join(sentence.split()[:15])
                if len(claim) > 20:
                    potential_claims.append(claim)
        
        return potential_claims[:max_phrases]
    
    def format_fact_check_results(self, claims: List[Dict]) -> List[Dict]:
        """Format fact check results for display"""
        formatted = []
        
        for claim in claims[:5]:
            try:
                claim_text = claim.get('text', 'N/A')
                claimant = claim.get('claimant', 'Unknown')
                claim_date = claim.get('claimDate', 'Unknown date')
                reviews = claim.get('claimReview', [])
                
                for review in reviews[:2]:
                    publisher = review.get('publisher', {}).get('name', 'Unknown')
                    rating = review.get('textualRating', 'Not rated')
                    title = review.get('title', 'No title')
                    url = review.get('url', '#')
                    
                    formatted.append({
                        'claim': claim_text,
                        'claimant': claimant,
                        'date': claim_date,
                        'publisher': publisher,
                        'rating': rating,
                        'title': title,
                        'url': url
                    })
            except Exception:
                continue
        
        return formatted

# ----------------------------
# Load Model and Vectorizer
# ----------------------------
@st.cache_resource
def load_model():
    try:
        with open("fake_news_model_v2.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer_v2.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run `train_model.py` first.")
        st.info("""
        **How to train the model:**
        1. Download the dataset from Kaggle
        2. Place Fake.csv and True.csv in the same directory
        3. Run: `python train_model.py`
        4. Then run: `streamlit run app.py`
        """)
        st.stop()

model, vectorizer = load_model()

# ----------------------------
# Text Preprocessing
# ----------------------------
def preprocess_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    return text

# ----------------------------
# Check for Trusted Sources
# ----------------------------
def check_trusted_source(text):
    """Check if text mentions trusted news sources"""
    trusted_sources = [
        "bbc", "reuters", "associated press", "apnews", "nasa", "nytimes", 
        "new york times", "guardian", "cnn", "washington post", "washingtonpost",
        "forbes", "bloomberg", "wall street journal", "wsj", "npr", "pbs",
        "abc news", "nbc news", "cbs news", "the economist", "time magazine"
    ]
    text_lower = text.lower()
    return any(src in text_lower for src in trusted_sources)

# ----------------------------
# Analyze Risk Factors
# ----------------------------
def analyze_risk_factors(text):
    """Identify potential red flags in the text"""
    warnings = []
    
    exclamation_count = text.count("!")
    if exclamation_count > 3:
        warnings.append(f"⚠️ Excessive exclamation marks ({exclamation_count})")
    
    clickbait_words = ["shocking", "breaking", "urgent", "unbelievable", "you won't believe", 
                       "doctors hate", "this one trick", "what happens next"]
    found_clickbait = [word for word in clickbait_words if word in text.lower()]
    if found_clickbait:
        warnings.append(f"⚠️ Clickbait keywords: {', '.join(found_clickbait)}")
    
    viral_phrases = ["share this", "share now", "spread the word", "before it's deleted"]
    found_viral = [phrase for phrase in viral_phrases if phrase in text.lower()]
    if found_viral:
        warnings.append(f"⚠️ Viral call-to-action: {', '.join(found_viral)}")
    
    word_count = len(text.split())
    if word_count < 50:
        warnings.append(f"⚠️ Very short content ({word_count} words)")
    
    if any(word.isupper() and len(word) > 3 for word in text.split()):
        warnings.append("⚠️ Excessive capitalization detected")
    
    if not warnings:
        warnings.append("✅ No major red flags detected")
    
    return warnings

# ----------------------------
# Prediction Function
# ----------------------------
def predict_news(news_text, confidence_threshold=0.55):
    """Predict if news is fake or real"""
    clean_text = preprocess_text(news_text)
    
    if len(clean_text.strip()) == 0:
        return None, None, False, "uncertain"
    
    tfidf = vectorizer.transform([clean_text])
    proba = model.predict_proba(tfidf)[0]
    
    fake_prob = proba[0]
    real_prob = proba[1]
    
    trusted = check_trusted_source(news_text)
    
    if trusted:
        boost = 0.25 if real_prob >= 0.40 else 0.20
        real_prob = min(real_prob + boost, 0.98)
        fake_prob = max(1.0 - real_prob, 0.02)
    
    authoritative_keywords = ['nasa', 'study', 'research', 'university', 'scientists', 
                              'according to', 'officials', 'data', 'announced']
    has_authority = sum(1 for kw in authoritative_keywords if kw in news_text.lower()) >= 2
    
    if has_authority and real_prob >= 0.35:
        real_prob = min(real_prob + 0.15, 0.98)
        fake_prob = max(1.0 - real_prob, 0.02)
    
    max_prob = max(fake_prob, real_prob)
    
    if max_prob < confidence_threshold:
        prediction = "uncertain"
        label = None
    elif real_prob > fake_prob:
        prediction = "real"
        label = 1
    else:
        prediction = "fake"
        label = 0
    
    return label, (fake_prob, real_prob), trusted, prediction

# ----------------------------
# Main UI
# ----------------------------
st.markdown("<h1 class='main-header'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-powered fake news detection with Google Fact Check verification</p>", unsafe_allow_html=True)

st.write("")

# API Key Configuration in Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Load from .env first
    if env_api_key:
        st.success("✅ API Key loaded from .env file")
        st.caption(f"Key: {env_api_key[:10]}...")
        api_key = None
    else:
        api_key = st.text_input(
            "Google Fact Check API Key (Optional)",
            type="password",
            help="Get your free API key from: https://console.cloud.google.com/"
        )
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.info("ℹ️ Add API key to enable fact checking")
    
    with st.expander("🔑 How to get API Key"):
        st.markdown("""
        **Steps to get Google Fact Check API Key:**
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project (or select existing)
        3. Enable "Fact Check Tools API"
        4. Go to "Credentials" → "Create Credentials" → "API Key"
        5. Copy and paste the key above
        
        **Note:** The API is completely free with no billing required!
        """)

# Initialize Fact Check API with correct priority
effective_api_key = env_api_key if env_api_key else api_key
fact_checker = FactCheckAPI(effective_api_key)

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📝 Full Article", "🔍 Quick Check"])

with tab1:
    st.write("### Enter Complete News Article")
    news_title = st.text_input("**News Title**", placeholder="Enter the headline or title")
    news_content = st.text_area("**News Content**", height=250, 
                                placeholder="Paste the full content here...")

with tab2:
    st.write("### Quick Text Analysis")
    quick_text = st.text_area("**Paste any news text**", height=200,
                             placeholder="Paste any news snippet...")

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze News", use_container_width=True)

if analyze_button:
    if news_title or news_content:
        full_text = f"{news_title} {news_content}".strip()
    else:
        full_text = quick_text.strip()
    
    if not full_text:
        st.warning("⚠️ Please enter some news content to analyze.")
    else:
        with st.spinner("🔄 Analyzing content..."):
            pred, proba, trusted, prediction_type = predict_news(full_text, confidence_threshold=0.60)
            
            if pred is None and prediction_type != "uncertain":
                st.error("❌ Unable to analyze the text. Please enter valid content.")
            else:
                fake_prob = proba[0] * 100
                real_prob = proba[1] * 100
                
                st.write("---")
                
                # Main Result
                st.write("## 📊 AI Model Analysis")
                
                if prediction_type == "uncertain":
                    st.warning(f"""
                    ### ⚠️ **UNCERTAIN PREDICTION**
                    The model cannot confidently classify this content.
                    
                    **Confidence levels:**
                    - Real: {real_prob:.1f}%
                    - Fake: {fake_prob:.1f}%
                    """)
                elif prediction_type == "real":
                    st.success(f"""
                    ### ✅ This appears to be **REAL NEWS**
                    **Confidence: {real_prob:.1f}%**
                    """)
                else:
                    st.error(f"""
                    ### 🚨 This appears to be **FAKE or MISLEADING**
                    **Confidence: {fake_prob:.1f}%**
                    """)
                
                if trusted:
                    st.info("ℹ️ **Trusted source detected** - confidence adjusted (+20-25%)")
                
                # Confidence metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fake Probability", f"{fake_prob:.1f}%")
                    st.progress(fake_prob / 100)
                with col2:
                    st.metric("Real Probability", f"{real_prob:.1f}%")
                    st.progress(real_prob / 100)
                
                # Google Fact Check Results
                st.write("---")
                st.write("## 🔍 Google Fact Check Verification")
                
                if fact_checker.api_key:
                    with st.spinner("🌐 Checking with Google Fact Check API..."):
                        key_phrases = fact_checker.extract_key_phrases(full_text)
                        
                        if key_phrases:
                            st.info(f"🔎 Searching for fact checks on {len(key_phrases)} key claims...")
                            
                            all_results = []
                            for phrase in key_phrases:
                                claims = fact_checker.search_claims(phrase)
                                if claims:
                                    formatted = fact_checker.format_fact_check_results(claims)
                                    all_results.extend(formatted)
                            
                            if all_results:
                                st.success(f"✅ Found {len(all_results)} related fact checks!")
                                
                                for i, result in enumerate(all_results[:5], 1):
                                    st.markdown(f"""
                                    <div class='claim-item'>
                                        <h4>Fact Check #{i}</h4>
                                        <p><strong>Claim:</strong> {result['claim'][:200]}...</p>
                                        <p><strong>Claimant:</strong> {result['claimant']} ({result['date']})</p>
                                        <p><strong>Rating:</strong> <span style='color: #1f77b4; font-weight: bold;'>{result['rating']}</span></p>
                                        <p><strong>Verified by:</strong> {result['publisher']}</p>
                                        <p><a href="{result['url']}" target="_blank">📄 Read full fact check →</a></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("ℹ️ No related fact checks found in Google's database. This doesn't mean the content is false - it may simply not have been fact-checked yet.")
                        else:
                            st.info("ℹ️ Could not extract specific claims for fact checking.")
                else:
                    st.warning("⚠️ No API key found")
                    st.info("""
                    **To enable fact checking:**
                    1. Create a `.env` file in the project directory
                    2. Add: `FACTCHECK_API_KEY=your_api_key_here`
                    3. Get free API key from: https://console.cloud.google.com/
                    4. Restart the Streamlit app
                    
                    Or enter your API key in the sidebar above.
                    """)
                
                # Risk Analysis
                st.write("---")
                st.write("## 🔍 Content Analysis")
                risk_factors = analyze_risk_factors(full_text)
                for factor in risk_factors:
                    st.write(factor)
                
                # Content Statistics
                st.write("## 📈 Content Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Word Count", len(full_text.split()))
                with col2:
                    st.metric("Character Count", len(full_text))
                with col3:
                    st.metric("Exclamation Marks", full_text.count("!"))
                with col4:
                    st.metric("Question Marks", full_text.count("?"))
                
                st.write("---")
                st.caption(f"🕒 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Tips
                with st.expander("💡 Tips for Verifying News"):
                    st.write("""
                    **How to spot fake news:**
                    1. ✓ Check multiple trusted sources (BBC, Reuters, AP)
                    2. ✓ Verify author credentials and publication date
                    3. ✓ Look for emotional manipulation and clickbait
                    4. ✓ Use reverse image search for photos
                    5. ✓ Check fact-checking websites (Snopes, FactCheck.org)
                    6. ✓ Verify the URL is legitimate
                    7. ✓ Look for credible expert quotes
                    8. ✓ Check the "About Us" page
                    9. ✓ Be skeptical of outrageous claims
                    10. ✓ Use Google Fact Check Explorer
                    """)

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.title("ℹ️ About This App")
st.sidebar.info("""
**Enhanced Fake News Detection System**

Combines machine learning with Google Fact Check API for comprehensive verification.

**Features:**
- AI Model Analysis (98% accuracy)
- Google Fact Check Integration
- Trusted source verification
- Content risk analysis
- Confidence scoring

**Model Details:**
- Algorithm: Logistic Regression
- Training: 40,000+ articles
- TF-IDF features (uni-grams & bi-grams)
""")

st.sidebar.markdown("---")
st.sidebar.write("**Dataset:** [Kaggle Fake & Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)")
st.sidebar.write("👨‍💻 **Course:** Cloud Computing (BITE412L)")
st.sidebar.write("🔬 **Version:** 3.0 (Enhanced)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool provides automated predictions. 
    Always verify with multiple sources and use fact-checking websites.</p>
</div>
""", unsafe_allow_html=True) '''

'working next'
import streamlit as st
import pickle
import re
import os
import string
from datetime import datetime
import numpy as np
import requests
from typing import List, Dict, Optional

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-top: 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .fact-check-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
        background-color: #f0f8ff;
    }
    .claim-item {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background-color: white;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Google Fact Check API Integration
# ----------------------------

from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Get your API key
env_api_key = os.getenv("FACTCHECK_API_KEY")

class FactCheckAPI:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    def search_claims(self, query: str, language: str = "en") -> List[Dict]:
        """Search for fact-checked claims related to the query"""
        if not self.api_key:
            return []
        
        try:
            params = {
                'query': query[:200],  # Limit query length
                'languageCode': language,
                'key': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('claims', [])
            else:
                st.warning(f"Fact Check API returned status code: {response.status_code}")
                return []
        except requests.exceptions.Timeout:
            st.warning("⏱️ Fact check API request timed out")
            return []
        except Exception as e:
            st.warning(f"⚠️ Fact check API error: {str(e)}")
            return []
    
    def extract_key_phrases(self, text: str, max_phrases: int = 3) -> List[str]:
        """Extract key phrases from text for fact checking"""
        # Remove common words and extract meaningful phrases
        text = text.lower()
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Extract potential claims (sentences with specific keywords)
        claim_keywords = ['announced', 'revealed', 'discovered', 'confirmed', 
                         'reported', 'said', 'stated', 'claimed', 'found']
        
        potential_claims = []
        for sentence in sentences[:10]:  # Check first 10 sentences
            if any(keyword in sentence for keyword in claim_keywords):
                # Clean and truncate
                claim = ' '.join(sentence.split()[:15])  # First 15 words
                if len(claim) > 20:  # Minimum length
                    potential_claims.append(claim)
        
        return potential_claims[:max_phrases]
    
    def format_fact_check_results(self, claims: List[Dict]) -> List[Dict]:
        """Format fact check results for display"""
        formatted = []
        
        for claim in claims[:5]:  # Limit to 5 results
            try:
                claim_text = claim.get('text', 'N/A')
                claimant = claim.get('claimant', 'Unknown')
                claim_date = claim.get('claimDate', 'Unknown date')
                
                # Get claim reviews
                reviews = claim.get('claimReview', [])
                
                for review in reviews[:2]:  # Limit to 2 reviews per claim
                    publisher = review.get('publisher', {}).get('name', 'Unknown')
                    rating = review.get('textualRating', 'Not rated')
                    title = review.get('title', 'No title')
                    url = review.get('url', '#')
                    
                    formatted.append({
                        'claim': claim_text,
                        'claimant': claimant,
                        'date': claim_date,
                        'publisher': publisher,
                        'rating': rating,
                        'title': title,
                        'url': url
                    })
            except Exception as e:
                continue
        
        return formatted

# ----------------------------
# Load Model and Vectorizer
# ----------------------------
@st.cache_resource
def load_model():
    try:
        with open("fake_news_model_v2.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer_v2.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run `train_model.py` first.")
        st.info("""
        **How to train the model:**
        1. Download the dataset from Kaggle: 
           https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
        2. Place Fake.csv and True.csv in the same directory
        3. Run: `python train_model.py`
        4. Then run: `streamlit run app.py`
        """)
        st.stop()

model, vectorizer = load_model()

# ----------------------------
# Text Preprocessing
# ----------------------------
def preprocess_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    return text

# ----------------------------
# Check for Trusted Sources
# ----------------------------
def check_trusted_source(text):
    """Check if text mentions trusted news sources"""
    trusted_sources = [
        "bbc", "reuters", "associated press", "apnews", "nasa", "nytimes", 
        "new york times", "guardian", "cnn", "washington post", "washingtonpost",
        "forbes", "bloomberg", "wall street journal", "wsj", "npr", "pbs",
        "abc news", "nbc news", "cbs news", "the economist", "time magazine"
    ]
    text_lower = text.lower()
    return any(src in text_lower for src in trusted_sources)

# ----------------------------
# Analyze Risk Factors
# ----------------------------
def analyze_risk_factors(text):
    """Identify potential red flags in the text"""
    warnings = []
    
    exclamation_count = text.count("!")
    if exclamation_count > 3:
        warnings.append(f"⚠️ Excessive exclamation marks ({exclamation_count})")
    
    clickbait_words = ["shocking", "breaking", "urgent", "unbelievable", "you won't believe", 
                       "doctors hate", "this one trick", "what happens next"]
    found_clickbait = [word for word in clickbait_words if word in text.lower()]
    if found_clickbait:
        warnings.append(f"⚠️ Clickbait keywords: {', '.join(found_clickbait)}")
    
    viral_phrases = ["share this", "share now", "spread the word", "before it's deleted"]
    found_viral = [phrase for phrase in viral_phrases if phrase in text.lower()]
    if found_viral:
        warnings.append(f"⚠️ Viral call-to-action: {', '.join(found_viral)}")
    
    word_count = len(text.split())
    if word_count < 50:
        warnings.append(f"⚠️ Very short content ({word_count} words)")
    
    if any(word.isupper() and len(word) > 3 for word in text.split()):
        warnings.append("⚠️ Excessive capitalization detected")
    
    if not warnings:
        warnings.append("✅ No major red flags detected")
    
    return warnings

# ----------------------------
# Prediction Function
# ----------------------------
def predict_news(news_text, confidence_threshold=0.55):
    """Predict if news is fake or real"""
    clean_text = preprocess_text(news_text)
    
    if len(clean_text.strip()) == 0:
        return None, None, False, "uncertain"
    
    tfidf = vectorizer.transform([clean_text])
    proba = model.predict_proba(tfidf)[0]
    
    fake_prob = proba[0]
    real_prob = proba[1]
    
    trusted = check_trusted_source(news_text)
    
    if trusted:
        if real_prob >= 0.40:
            boost = 0.25
        else:
            boost = 0.20
        real_prob = min(real_prob + boost, 0.98)
        fake_prob = max(1.0 - real_prob, 0.02)
    
    authoritative_keywords = ['nasa', 'study', 'research', 'university', 'scientists', 
                              'according to', 'officials', 'data', 'announced']
    has_authority = sum(1 for kw in authoritative_keywords if kw in news_text.lower()) >= 2
    
    if has_authority and real_prob >= 0.35:
        real_prob = min(real_prob + 0.15, 0.98)
        fake_prob = max(1.0 - real_prob, 0.02)
    
    max_prob = max(fake_prob, real_prob)
    
    if max_prob < confidence_threshold:
        prediction = "uncertain"
        label = None
    elif real_prob > fake_prob:
        prediction = "real"
        label = 1
    else:
        prediction = "fake"
        label = 0
    
    return label, (fake_prob, real_prob), trusted, prediction

# ----------------------------
# Main UI
# ----------------------------
st.markdown("<h1 class='main-header'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-powered fake news detection with Google Fact Check verification</p>", unsafe_allow_html=True)

st.write("")

# API Key Configuration in Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Load from .env first
    if env_api_key:
        st.success("✅ API Key loaded from .env file")
        st.caption(f"Key: {env_api_key[:10]}...")
        api_key = None  # Don't need sidebar input
    else:
        api_key = st.text_input(
            "Google Fact Check API Key (Optional)",
            type="password",
            help="Get your free API key from: https://console.cloud.google.com/"
        )
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.info("ℹ️ Add API key to enable fact checking")
    
    with st.expander("🔑 How to get API Key"):
        st.markdown("""
        **Steps to get Google Fact Check API Key:**
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project (or select existing)
        3. Enable "Fact Check Tools API"
        4. Go to "Credentials" → "Create Credentials" → "API Key"
        5. Copy and paste the key above
        
        **Note:** The API is completely free with no billing required!
        """)

# Initialize Fact Check API
fact_checker = FactCheckAPI(api_key if api_key else None)

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📝 Full Article", "🔍 Quick Check"])

with tab1:
    st.write("### Enter Complete News Article")
    news_title = st.text_input("**News Title**", placeholder="Enter the headline or title")
    news_content = st.text_area("**News Content**", height=250, 
                                placeholder="Paste the full content here...")

with tab2:
    st.write("### Quick Text Analysis")
    quick_text = st.text_area("**Paste any news text**", height=200,
                             placeholder="Paste any news snippet...")

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze News", use_container_width=True)

if analyze_button:
    # Determine which input to use
    if news_title or news_content:
        full_text = f"{news_title} {news_content}".strip()
    else:
        full_text = quick_text.strip()
    
    if not full_text:
        st.warning("⚠️ Please enter some news content to analyze.")
    else:
        with st.spinner("🔄 Analyzing content..."):
            pred, proba, trusted, prediction_type = predict_news(full_text, confidence_threshold=0.60)
            
            if pred is None and prediction_type != "uncertain":
                st.error("❌ Unable to analyze the text. Please enter valid content.")
            else:
                fake_prob = proba[0] * 100
                real_prob = proba[1] * 100
                
                st.write("---")
                
                # Main Result
                st.write("## 📊 AI Model Analysis")
                
                if prediction_type == "uncertain":
                    st.warning(f"""
                    ### ⚠️ **UNCERTAIN PREDICTION**
                    The model cannot confidently classify this content.
                    
                    **Confidence levels:**
                    - Real: {real_prob:.1f}%
                    - Fake: {fake_prob:.1f}%
                    """)
                elif prediction_type == "real":
                    st.success(f"""
                    ### ✅ This appears to be **REAL NEWS**
                    **Confidence: {real_prob:.1f}%**
                    """)
                else:
                    st.error(f"""
                    ### 🚨 This appears to be **FAKE or MISLEADING**
                    **Confidence: {fake_prob:.1f}%**
                    """)
                
                if trusted:
                    st.info("ℹ️ **Trusted source detected** - confidence adjusted (+20-25%)")
                
                # Confidence metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fake Probability", f"{fake_prob:.1f}%")
                    st.progress(fake_prob / 100)
                with col2:
                    st.metric("Real Probability", f"{real_prob:.1f}%")
                    st.progress(real_prob / 100)
                
                # Google Fact Check Results
                st.write("---")
                st.write("## 🔍 Google Fact Check Verification")
                
                if fact_checker.api_key:
                    with st.spinner("🌐 Checking with Google Fact Check API..."):
                        # Extract key phrases for fact checking
                        key_phrases = fact_checker.extract_key_phrases(full_text)
                        
                        if key_phrases:
                            st.info(f"🔎 Searching for fact checks on {len(key_phrases)} key claims...")
                            
                            all_results = []
                            for phrase in key_phrases:
                                claims = fact_checker.search_claims(phrase)
                                if claims:
                                    formatted = fact_checker.format_fact_check_results(claims)
                                    all_results.extend(formatted)
                            
                            if all_results:
                                st.success(f"✅ Found {len(all_results)} related fact checks!")
                                
                                for i, result in enumerate(all_results[:5], 1):
                                    with st.container():
                                        st.markdown(f"""
                                        <div class='claim-item'>
                                            <h4>Fact Check #{i}</h4>
                                            <p><strong>Claim:</strong> {result['claim'][:200]}...</p>
                                            <p><strong>Claimant:</strong> {result['claimant']} ({result['date']})</p>
                                            <p><strong>Rating:</strong> <span style='color: #1f77b4; font-weight: bold;'>{result['rating']}</span></p>
                                            <p><strong>Verified by:</strong> {result['publisher']}</p>
                                            <p><a href="{result['url']}" target="_blank">📄 Read full fact check →</a></p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info("ℹ️ No related fact checks found in Google's database. This doesn't mean the content is false - it may simply not have been fact-checked yet.")
                        else:
                            st.info("ℹ️ Could not extract specific claims for fact checking.")
                else:
                    st.warning("⚠️ No API key found in .env file")
                    st.info("""
                    **To enable fact checking:**
                    1. Create a `.env` file in the project directory
                    2. Add: `FACTCHECK_API_KEY=your_api_key_here`
                    3. Get free API key from: https://console.cloud.google.com/
                    4. Restart the Streamlit app
                    
                    The Fact Check API is completely FREE with no billing required!
                    """)
                
                # Risk Analysis
                st.write("---")
                st.write("## 🔍 Content Analysis")
                risk_factors = analyze_risk_factors(full_text)
                for factor in risk_factors:
                    st.write(factor)
                
                # Content Statistics
                st.write("## 📈 Content Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Word Count", len(full_text.split()))
                with col2:
                    st.metric("Character Count", len(full_text))
                with col3:
                    st.metric("Exclamation Marks", full_text.count("!"))
                with col4:
                    st.metric("Question Marks", full_text.count("?"))
                
                st.write("---")
                st.caption(f"🕒 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Tips
                with st.expander("💡 Tips for Verifying News"):
                    st.write("""
                    **How to spot fake news:**
                    1. ✓ Check multiple trusted sources (BBC, Reuters, AP)
                    2. ✓ Verify author credentials and publication date
                    3. ✓ Look for emotional manipulation and clickbait
                    4. ✓ Use reverse image search for photos
                    5. ✓ Check fact-checking websites (Snopes, FactCheck.org)
                    6. ✓ Verify the URL is legitimate
                    7. ✓ Look for credible expert quotes
                    8. ✓ Check the "About Us" page
                    9. ✓ Be skeptical of outrageous claims
                    10. ✓ Use Google Fact Check Explorer
                    """)

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.title("ℹ️ About This App")
st.sidebar.info("""
**Enhanced Fake News Detection System**

Combines machine learning with Google Fact Check API for comprehensive verification.

**Features:**
- AI Model Analysis (98% accuracy)
- Google Fact Check Integration
- Trusted source verification
- Content risk analysis
- Confidence scoring

**Model Details:**
- Algorithm: Logistic Regression
- Training: 40,000+ articles
- TF-IDF features (uni-grams & bi-grams)
""")

st.sidebar.markdown("---")
st.sidebar.write("**Dataset:** [Kaggle Fake & Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)")
st.sidebar.write("👨‍💻 **Course:** Cloud Computing (BITE412L)")
st.sidebar.write("🔬 **Version:** 3.0 (Enhanced)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool provides automated predictions. 
    Always verify with multiple sources and use fact-checking websites.</p>
</div>
""", unsafe_allow_html=True)
