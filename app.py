
import streamlit as st
import pickle
import re
import string
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import time

# Page Configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #0078D7, #00BFA6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #444;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* ===== DETECTED RESULT BOXES ===== */
    .fake-news {
        background-color: #ffe5e5;
        color: #c62828;
        padding: 28px;
        border-radius: 14px;
        border-left: 6px solid #f44336;
        box-shadow: 0 6px 12px rgba(244,67,54,0.25);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .fake-news h2 {
        color: #c62828;
        margin: 0;
    }
    .fake-news p {
        color: #c62828;
        margin: 8px 0 0 0;
    }
    .real-news {
        background-color: #e3fbe3;
        color: #2e7d32;
        padding: 28px;
        border-radius: 14px;
        border-left: 6px solid #4caf50;
        box-shadow: 0 6px 12px rgba(76,175,80,0.25);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .real-news h2 {
        color: #2e7d32;
        margin: 0;
    }
    .real-news p {
        color: #2e7d32;
        margin: 8px 0 0 0;
    }

    /* ===== RELIABLE NEWS SOURCES ===== */
    .related-article {
        background-color: #f8f8f8;
        color: #333;
        padding: 14px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 3px solid #2196F3;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    .related-article:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .related-article h4 {
        font-size: 1rem;
        margin-bottom: 6px;
        color: #0a57d1;
        font-weight: 600;
    }
    .related-article p {
        margin: 4px 0;
        color: #555;
        line-height: 1.5;
    }
    .related-article .source-info {
        font-size: 0.85rem;
        color: #777;
        margin-top: 6px;
    }

    /* ===== FACT CHECK / RISK ANALYSIS ===== */
    .fact-check {
        background-color: #fff0cc;
        color: #5a3e00;
        padding: 18px 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        box-shadow: 0 3px 8px rgba(255,152,0,0.25);
    }

    h2 {
        font-weight: 800;
    }
    a {
        color: #0078D7;
        text-decoration: none;
        font-weight: 600;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load model and vectorizer"""
    try:
        with open('fake_news_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Error: Model file not found! Please run train_model.py first to train the model.")
        st.stop()

def preprocess_text(text):
    """Text preprocessing"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

def predict_news(text, model, vectorizer):
    """Predict whether news is real or fake"""
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    return prediction, probability

def extract_keywords(text, num_keywords=5):
    """Extract keywords from the text"""
    common_words = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','up','about','into','through','during','is','are','was','were','be','been','being','have','has','had','do','does','did','will','would','should','could','may','might','must','can','this','that','these','those','i','you','he','she','it','we','they','them','their','what','which','who','when','where','why','how','all','each','every','both','few','more','most','other','some','such','only','own','same','so','than','too','very','said','says','just','also'}
    words = preprocess_text(text).split()
    word_freq = {}
    for word in words:
        if word not in common_words and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:num_keywords]]

def search_related_news(keywords, title):
    """Search for related news articles from reliable sources"""
    search_query = " ".join(keywords[:3])
    articles = []
    
    try:
        # Use DuckDuckGo HTML search
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query + ' news')}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.find_all('div', class_='result', limit=8)
            
            reliable_domains = ['reuters.com', 'apnews.com', 'bbc.com', 'npr.org', 
                              'theguardian.com', 'nytimes.com', 'washingtonpost.com',
                              'cnn.com', 'abcnews.go.com', 'cbsnews.com']
            
            for result in results:
                try:
                    title_elem = result.find('a', class_='result__a')
                    snippet_elem = result.find('a', class_='result__snippet')
                    
                    if title_elem and snippet_elem:
                        article_title = title_elem.get_text(strip=True)
                        article_url = title_elem.get('href', '')
                        article_snippet = snippet_elem.get_text(strip=True)
                        
                        # Check if from reliable source
                        is_reliable = any(domain in article_url.lower() for domain in reliable_domains)
                        
                        if article_url and article_title:
                            # Extract source name from URL
                            source_name = "Unknown Source"
                            for domain in reliable_domains:
                                if domain in article_url.lower():
                                    source_name = domain.split('.')[0].upper()
                                    if source_name == 'APNEWS':
                                        source_name = 'AP News'
                                    elif source_name == 'NPR':
                                        source_name = 'NPR'
                                    elif source_name == 'THEGUARDIAN':
                                        source_name = 'The Guardian'
                                    elif source_name == 'NYTIMES':
                                        source_name = 'NY Times'
                                    elif source_name == 'WASHINGTONPOST':
                                        source_name = 'Washington Post'
                                    break
                            
                            articles.append({
                                'title': article_title,
                                'url': article_url,
                                'snippet': article_snippet[:200] + '...' if len(article_snippet) > 200 else article_snippet,
                                'source': source_name,
                                'is_reliable': is_reliable
                            })
                            
                            if len(articles) >= 5:
                                break
                except:
                    continue
                    
    except Exception as e:
        st.warning(f"Unable to retrieve real-time news articles. Please search for keywords manually.：{search_query}")
    
    # Fallback: reliable source homepages
    reliable_sources = [
        {'name': 'Reuters', 'url': 'https://www.reuters.com', 'description': 'Global news agency known for objective reporting.'},
        {'name': 'Associated Press (AP)', 'url': 'https://apnews.com', 'description': 'Independent, trusted news source worldwide.'},
        {'name': 'BBC News', 'url': 'https://www.bbc.com/news', 'description': 'Renowned British news network.'},
        {'name': 'NPR', 'url': 'https://www.npr.org', 'description': 'National Public Radio - U.S. public news outlet.'},
        {'name': 'The Guardian', 'url': 'https://www.theguardian.com', 'description': 'Award-winning British media organization.'}
    ]
    
    fact_check_sites = [
        {'name': 'Snopes', 'url': 'https://www.snopes.com', 'description': 'One of the oldest and most respected fact-checking sites.'},
        {'name': 'FactCheck.org', 'url': 'https://www.factcheck.org', 'description': 'Independent fact-checking organization.'},
        {'name': 'PolitiFact', 'url': 'https://www.politifact.com', 'description': 'Pulitzer Prize-winning political fact-checking platform.'},
        {'name': 'Full Fact', 'url': 'https://fullfact.org', 'description': 'Independent UK fact-checking charity.'}
    ]
    
    return {
        'search_query': search_query,
        'articles': articles,
        'reliable_sources': reliable_sources,
        'fact_check_sites': fact_check_sites,
        'keywords': keywords
    }

def generate_fact_check_tips(text):
    """Generate fact-checking advice"""
    tips = []
    text_lower = text.lower()
    if any(word in text_lower for word in ['secret','hidden','cover up','conspiracy']):
        tips.append("⚠️ Contains conspiracy-related terms — verify the reliability of sources.")
    if any(word in text_lower for word in ['miracle','cure all','scientists dont want','doctors hate']):
        tips.append("⚠️ Contains exaggerated medical claims — consult professional sources.")
    if 'anonymous' in text_lower:
        tips.append("⚠️ Mentions anonymous sources — verify information authenticity.")
    if any(word in text_lower for word in ['share before','share this','they dont want you','wake up']):
        tips.append("⚠️ Urges quick sharing — a common fake news tactic.")
    if text.count('!') > 5:
        tips.append("⚠️ Excessive exclamation marks — may indicate sensationalism.")
    if not any(char.isdigit() for char in text):
        tips.append("⚠️ Lacks numerical data — real news often includes verifiable stats.")
    return tips if tips else ["✓ No obvious signs of fake news detected."]

# Load model
model, vectorizer = load_model()

# Headers
st.markdown('<div class="main-header">🔍 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Identify fake news using machine learning | Reliable source suggestions included</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **Model Info:**
    - Algorithm: Logistic Regression  
    - Features: TF-IDF Vectorization  
    - Training Data: 44,000+ news articles  
    - Expected Accuracy: ~98%

    **New Features:**
    - ✅ Auto-suggests trusted news sources  
    - ✅ Provides fact-check websites  
    - ✅ Generates search recommendations  
    - ✅ Smart risk analysis  
    - ✅ Real news article links
    """)
    st.header("📊 Statistics")
    if 'detection_count' not in st.session_state:
        st.session_state.detection_count = 0
    if 'fake_count' not in st.session_state:
        st.session_state.fake_count = 0
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Total Checks", st.session_state.detection_count)
    with col_stat2:
        st.metric("Fake News Found", st.session_state.fake_count)

# Main Section
col1, col2 = st.columns([2, 1])
with col1:
    st.header("📝 Enter News Content")
    news_title = st.text_input("News Title", placeholder="Enter the news headline...")
    news_text = st.text_area("News Body", height=250, placeholder="Paste or type the news content here...")

    with st.expander("📋 View Example News"):
        example_type = st.radio("Select Example Type", ["Real News", "Fake News"])
        if example_type == "Real News":
            st.info("""
            **Title:** Tech Giant Announces New AI Research Initiative

            **Body:** A major technology company announced today a new research initiative focused on artificial intelligence safety and ethics...
            """)
        else:
            st.warning("""
            **Title:** 5G Towers Confirmed to Control Human Thoughts, Secret Government Documents Reveal

            **Body:** Leaked documents claim 5G towers transmit mind-control signals. Scientists attempting to expose it have disappeared...
            """)
        if st.button("Use This Example"):
            if example_type == "Real News":
                st.session_state.example_title = "Tech Giant Announces New AI Research Initiative"
                st.session_state.example_text = "A major technology company announced today a new research initiative..."
            else:
                st.session_state.example_title = "5G Towers Confirmed to Control Human Thoughts, Secret Government Documents Reveal"
                st.session_state.example_text = "Leaked documents claim 5G towers transmit mind-control signals..."
            st.rerun()

    if 'example_title' in st.session_state:
        news_title = st.session_state.example_title
        news_text = st.session_state.example_text
        del st.session_state.example_title
        del st.session_state.example_text

with col2:
    st.header("🎯 Detection Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.05,
                                     help="Minimum confidence required for the prediction.")
    show_details = st.checkbox("Show Detailed Analysis", value=True)
    show_related = st.checkbox("Show Related Sources (for fake news)", value=True)

st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    detect_button = st.button("🔍 Start Detection", type="primary", use_container_width=True)

if detect_button:
    if not news_title and not news_text:
        st.error("⚠️ Please enter at least a title or body content!")
    else:
        with st.spinner("Analyzing news content..."):
            full_text = f"{news_title} {news_text}"
            prediction, probability = predict_news(full_text, model, vectorizer)
            st.session_state.detection_count += 1
            if prediction == 0:
                st.session_state.fake_count += 1
            st.markdown("---")
            st.header("📋 Detection Results")
            fake_prob = probability[0] * 100
            real_prob = probability[1] * 100

            if prediction == 1:
                st.markdown(f"""
                <div class="real-news">
                    <h2>✅ This is likely <strong>REAL NEWS</strong></h2>
                    <p style="font-size: 1.2rem;">Confidence: <strong>{real_prob:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(real_prob / 100)
                st.success("✓ The news appears genuine, but cross-check with multiple sources.")
            else:
                st.markdown(f"""
                <div class="fake-news">
                    <h2>⚠️ This is likely <strong>FAKE NEWS</strong></h2>
                    <p style="font-size: 1.2rem;">Confidence: <strong>{fake_prob:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(fake_prob / 100)
                st.error("⚠️ This content shows traits of fake news. Verify through trusted outlets!")
                st.markdown("---")
                st.subheader("🔍 Risk Analysis")
                fact_check_tips = generate_fact_check_tips(full_text)
                st.markdown('<div class="fact-check">', unsafe_allow_html=True)
                st.markdown("**Detected Risk Features:**")
                for tip in fact_check_tips:
                    st.markdown(f"- {tip}")
                st.markdown('</div>', unsafe_allow_html=True)
                if show_related:
                    st.markdown("---")
                    st.header("📰 Trusted Information Sources")
                    keywords = extract_keywords(full_text)
                    
                    with st.spinner("Searching for related news articles"):
                        related_info = search_related_news(keywords, news_title)
                    
                    st.info(f"**Suggested Search Keywords:** `{related_info['search_query']}`")
                    
                    # Display found articles
                    if related_info['articles']:
                        st.subheader("📰 Related News Articles from Trusted Sources")
                        for article in related_info['articles']:
                            reliability_badge = "✅ Reliable Source" if article['is_reliable'] else "ℹ️ General Source"
                            st.markdown(f"""
                            <div class="related-article">
                                <h4>📄 {article['title']}</h4>
                                <p>{article['snippet']}</p>
                                <p class="source-info"><strong>Source:</strong> {article['source']} | {reliability_badge}</p>
                                <p><a href="{article['url']}" target="_blank">Read Full Article →</a></p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No related articles found. Please try searching the suggested keywords manually.")
                    
                    st.subheader("✅ Reliable News Sources")
                    for source in related_info['reliable_sources']:
                        st.markdown(f"""
                        <div class="related-article">
                            <h4>🌐 {source['name']}</h4>
                            <p>{source['description']}</p>
                            <p><a href="{source['url']}" target="_blank">Visit Site →</a></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.subheader("🔎 Fact-Checking Websites")
                    for site in related_info['fact_check_sites']:
                        st.markdown(f"""
                        <div class="related-article">
                            <h4>✓ {site['name']}</h4>
                            <p>{site['description']}</p>
                            <p><a href="{site['url']}" target="_blank">Visit Site →</a></p>
                        </div>
                        """, unsafe_allow_html=True)
                    st.subheader("🔍 How to Verify Information")
                    st.markdown("""
                    **Recommended Steps:**
                    1. Search keywords on reliable sources above  
                    2. Check specific claims using fact-check sites  
                    3. Confirm with independent reports  
                    4. Review official data or statements  
                    5. Pay attention to the date and context  

                    **Warning Signs:**
                    - ❌ Only one outlet reports it  
                    - ❌ Anonymous or unclear source  
                    - ❌ Overly emotional headlines  
                    - ❌ Missing data or specifics  
                    - ❌ Urges immediate sharing  
                    """)

            if show_details:
                st.markdown("---")
                st.subheader("📊 Detailed Analysis")
                col_det1, col_det2 = st.columns(2)
                with col_det1:
                    st.metric("Fake News Probability", f"{fake_prob:.2f}%")
                with col_det2:
                    st.metric("Real News Probability", f"{real_prob:.2f}%")
                max_prob = max(fake_prob, real_prob)
                if max_prob >= confidence_threshold * 100:
                    st.success(f"✓ High model confidence ({max_prob:.2f}% ≥ {confidence_threshold*100}%)")
                else:
                    st.warning(f"⚠️ Low model confidence ({max_prob:.2f}% < {confidence_threshold*100}%), interpret cautiously.")
                word_count = len(full_text.split())
                char_count = len(full_text)
                st.info(f"""
                **Text Stats:**
                - Word Count: {word_count}
                - Character Count: {char_count}
                - Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for informational purposes only and should not replace human judgment. Always verify across multiple sources.</p>
    <p>🔬 Model trained using machine learning, estimated accuracy ~98%</p>
    <p>💡 When fake news is detected, the system recommends trusted news and fact-checking sites automatically.</p>
</div>
""", unsafe_allow_html=True)
