import streamlit as st
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import numpy as np
from dotenv import load_dotenv
import os
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Load environment variables
load_dotenv()

# Initialize the models
@st.cache_resource
def load_models():
    return {
        'semantic': SentenceTransformer('sentence-transformers/all-distilroberta-v1'),
        'tfidf': TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            stop_words='english'
        )
    }

models = load_models()

def get_search_results(query):
    """Get maximum possible search results using SerpAPI"""
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERPAPI_KEY"),
            "engine": "google",
            "num": 100,
            "gl": "us",
            "hl": "en"
        })
        
        results = search.get_dict()
        
        if "error" in results:
            st.error(f"SerpAPI Error: {results['error']}")
            return []
            
        if "organic_results" not in results:
            st.warning("No results found in the search.")
            return []
            
        return [(
            result.get("title", ""),
            result.get("snippet", ""),
            result.get("link", "")
        ) for result in results["organic_results"]]
        
    except Exception as e:
        st.error(f"Search API Error: {str(e)}")
        return []

def clean_text(text):
    """Clean and prepare text for comparison"""
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    # Preserve sentence endings while cleaning
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_sentences(text):
    """Split text into sentences using NLTK"""
    try:
        # Normalize periods to help NLTK
        text = re.sub(r'\.{2,}', '.', text)
        sentences = sent_tokenize(text)
        # Filter out very short sentences and normalize
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    except Exception:
        # Fallback to simple splitting if NLTK fails
        sentences = [s.strip() + '.' for s in text.split('.') if len(s.strip()) > 20]
        return sentences

def calculate_vectorized_similarity(input_sentences, source_sentences):
    """Calculate similarity between sentences"""
    if not input_sentences or not source_sentences:
        return {
            'similarity': 0,
            'matching_texts': ['', ''],
            'sentence_coverage': np.array([]),
            'sentence_similarities': np.array([]),
            'similarity_matrix': np.array([]),
            'good_matches': [],
            'all_matches': np.array([])
        }
    
    # Semantic similarity
    input_embeddings = models['semantic'].encode(input_sentences, convert_to_tensor=True)
    source_embeddings = models['semantic'].encode(source_sentences, convert_to_tensor=True)
    
    input_embeddings = input_embeddings.cpu().numpy()
    source_embeddings = source_embeddings.cpu().numpy()
    
    # Calculate semantic similarity matrix
    semantic_matrix = np.dot(input_embeddings, source_embeddings.T)
    
    # TF-IDF similarity
    tfidf_matrix = models['tfidf'].fit_transform(input_sentences + source_sentences)
    input_tfidf = tfidf_matrix[:len(input_sentences)]
    source_tfidf = tfidf_matrix[len(input_sentences):]
    tfidf_matrix = np.dot(input_tfidf.toarray(), source_tfidf.toarray().T)
    
    # Combined matrices with adjusted weights
    combined_matrix = 0.7 * semantic_matrix + 0.3 * tfidf_matrix
    
    # Lower threshold for better detection
    similarity_threshold = 0.4
    high_similarities = combined_matrix > similarity_threshold
    
    sentence_coverage = np.any(high_similarities, axis=1)
    sentence_max_similarities = np.max(combined_matrix, axis=1)
    
    max_similarity = np.max(combined_matrix)
    max_idx = np.unravel_index(np.argmax(combined_matrix), combined_matrix.shape)
    
    good_matches = []
    for i in range(len(input_sentences)):
        best_match_idx = np.argmax(combined_matrix[i])
        if combined_matrix[i][best_match_idx] > similarity_threshold:
            good_matches.append((i, best_match_idx, combined_matrix[i][best_match_idx]))
    
    return {
        'similarity': max_similarity,
        'matching_texts': [input_sentences[max_idx[0]], source_sentences[max_idx[1]]],
        'sentence_coverage': sentence_coverage,
        'sentence_similarities': sentence_max_similarities,
        'similarity_matrix': combined_matrix,
        'good_matches': good_matches,
        'all_matches': combined_matrix > similarity_threshold
    }

def analyze_internet_content(similarity_results, input_sentences):
    """Analyze what percentage of sentences appear to be from the internet"""
    if not input_sentences:
        return 0
        
    sentence_scores = np.zeros(len(input_sentences))
    match_counts = np.zeros(len(input_sentences))
    
    for result in similarity_results:
        sim_matrix = result.get('similarity_matrix', np.array([]))
        if sim_matrix.size > 0:
            sentence_max_sims = np.max(sim_matrix, axis=1)
            sentence_scores = np.maximum(sentence_scores, sentence_max_sims)
            match_counts += (sim_matrix > 0.4).sum(axis=1)
    
    # Calculate matched sentences with higher weights
    strong_matches = (sentence_scores > 0.7).sum()  # Increased threshold
    moderate_matches = (sentence_scores > 0.5).sum()
    weak_matches = (sentence_scores > 0.4).sum()
    
    total_sentences = len(input_sentences)
    
    # Weighted calculation
    base_percentage = (
        (strong_matches * 1.5 + 
         moderate_matches * 1.0 + 
         weak_matches * 0.5) / total_sentences
    ) * 100
    
    # Boost based on match frequency and coverage
    frequent_matches = (match_counts > 2).sum() / total_sentences
    coverage_boost = 1 + frequent_matches
    
    internet_percentage = min(base_percentage * coverage_boost * 1.3, 100)
    
    # Additional boost for high coverage
    if strong_matches / total_sentences > 0.6:
        internet_percentage = max(internet_percentage, 95)
    elif moderate_matches / total_sentences > 0.7:
        internet_percentage = max(internet_percentage, 85)
    
    return internet_percentage

@st.cache_data(ttl=3600)
def check_plagiarism(input_text, search_results):
    """Check plagiarism using sentence-based analysis"""
    plagiarism_results = []
    similarity_results = []
    
    # Clean and split input text into sentences
    cleaned_input = clean_text(input_text)
    input_sentences = split_into_sentences(cleaned_input)
    
    if not input_sentences:
        return [], 0, 0
    
    sentence_coverage = np.zeros(len(input_sentences), dtype=bool)
    sentence_max_similarities = np.zeros(len(input_sentences))
    
    total_sources = len(search_results)
    for idx, (title, snippet, link) in enumerate(search_results):
        try:
            progress = (idx + 1) / total_sources
            st.progress(progress, text=f"Analyzing source {idx + 1} of {total_sources}")
            
            source_text = clean_text(f"{title} {snippet}")
            source_sentences = split_into_sentences(source_text)
            
            if not source_sentences:
                continue
            
            similarity_result = calculate_vectorized_similarity(input_sentences, source_sentences)
            similarity_score = similarity_result['similarity'] * 100
            
            similarity_results.append(similarity_result)
            
            sentence_coverage = sentence_coverage | similarity_result['sentence_coverage']
            sentence_max_similarities = np.maximum(
                sentence_max_similarities, 
                similarity_result['sentence_similarities']
            )
            
            if similarity_score > 40:
                plagiarism_results.append({
                    'similarity': similarity_score,
                    'source_text': f"{title}\n{snippet}",
                    'link': link,
                    'matching_text': similarity_result['matching_texts']
                })
        except Exception as e:
            continue
    
    # Calculate internet percentage based on sentences
    internet_percentage = analyze_internet_content(similarity_results, input_sentences)
    
    # Calculate coverage with weighted sentence matching
    strong_matches = np.sum(sentence_max_similarities > 0.7)
    moderate_matches = np.sum(sentence_max_similarities > 0.5)
    weak_matches = np.sum(sentence_max_similarities > 0.4)
    
    total_sentences = len(input_sentences)
    coverage_percentage = (
        (strong_matches * 1.5 + 
         moderate_matches * 1.0 + 
         weak_matches * 0.5) / total_sentences
    ) * 100 * 1.3
    
    # Boost coverage based on internet content
    if internet_percentage > 80:
        coverage_percentage = max(coverage_percentage, internet_percentage * 1.1)
    
    if strong_matches / total_sentences > 0.6:
        coverage_percentage = max(coverage_percentage, 95)
    elif moderate_matches / total_sentences > 0.7:
        coverage_percentage = max(coverage_percentage, 85)
    
    coverage_percentage = min(coverage_percentage, 100)
    
    return plagiarism_results, coverage_percentage, internet_percentage

# Streamlit UI
st.title("📝 Advanced Plagiarism Detector")
st.markdown("*Using Sentence-Based Analysis*")

# API Status Check
api_key = os.getenv('SERPAPI_KEY')
if not api_key:
    st.error("⚠️ SERPAPI_KEY not found in .env file. Please configure your API key.")
    st.stop()

# Text input
input_text = st.text_area("Input Text", height=200)

# Advanced settings in sidebar
st.sidebar.title("Settings")
similarity_threshold = st.sidebar.number_input(
    "Similarity Threshold", 
    min_value=1, 
    max_value=100, 
    value=10,  
    help="Adjust the similarity threshold for plagiarism detection"
)

if st.button("🔍 Check Plagiarism"):
    if len(input_text) < 50:
        st.error("Please enter at least 50 characters for accurate plagiarism detection.")
    else:
        with st.spinner("🔄 Analyzing text for potential plagiarism..."):
            try:
                st.info("🌐 Searching for similar content...")
                search_results = get_search_results(input_text)
                
                if search_results:
                    st.info(f"Found {len(search_results)} sources to analyze...")
                    plagiarism_results, coverage_percentage, internet_percentage = check_plagiarism(
                        input_text, search_results
                    )
                    
                    if plagiarism_results:
                        filtered_results = [r for r in plagiarism_results if r['similarity'] >= similarity_threshold]
                        
                        if filtered_results:
                            st.warning("⚠️ Potential plagiarism detected!")
                            
                            # Display metrics
                            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                            with col1:
                                st.metric("Text Coverage", f"{coverage_percentage:.1f}%")
                            with col2:
                                st.metric("Internet Content", f"{internet_percentage:.1f}%")
                            with col3:
                                st.metric("Sources Analyzed", len(search_results))
                            with col4:
                                st.metric("Matches Found", len(filtered_results))
                            
                            # Analysis summary
                            st.info(f"""
                            📊 Analysis Summary:
                            - {coverage_percentage:.1f}% of your text matches other sources
                            - {internet_percentage:.1f}% of your text appears to be from the internet
                            - Found {len(filtered_results)} significant matches
                            - Analyzed {len(search_results)} potential sources
                            """)
                            
                            # Display results
                            filtered_results.sort(key=lambda x: x['similarity'], reverse=True)
                            for idx, result in enumerate(filtered_results, 1):
                                with st.expander(f"Match #{idx} - {result['similarity']:.1f}% Similar"):
                                    st.markdown("**Source Content:**")
                                    st.write(result['source_text'])
                                    st.markdown(f"**Source Link:** [{result['link']}]({result['link']})")
                                    
                                    if result.get('matching_text'):
                                        st.markdown("**Matching Text Portions:**")
                                        st.write("Your text: ", result['matching_text'][0])
                                        st.write("Source text: ", result['matching_text'][1])
                            
                            # Download report
                            if st.button("📥 Download Detailed Report"):
                                report = f"""Plagiarism Analysis Report
                                Overall Text Coverage: {coverage_percentage:.1f}%
                                Internet Content: {internet_percentage:.1f}%
                                Sources Analyzed: {len(search_results)}
                                Matches Found: {len(filtered_results)}
                                
                                Detailed Matches:
                                """
                                
                                report += "\n\n".join([
                                    f"Match #{i+1} ({r['similarity']:.1f}% Similar)\n"
                                    f"Source: {r['source_text']}\n"
                                    f"Link: {r['link']}\n"
                                    f"Matching Text:\n"
                                    f"Your text: {r['matching_text'][0]}\n"
                                    f"Source text: {r['matching_text'][1]}"
                                    for i, r in enumerate(filtered_results)
                                ])
                                
                                st.download_button(
                                    "📄 Save Report",
                                    report,
                                    "plagiarism_report.txt",
                                    "text/plain"
                                )
                        else:
                            st.success("✅ No significant plagiarism detected above the threshold!")
                    else:
                        st.success("✅ No significant plagiarism detected!")
                else:
                    st.warning("No search results found. Try modifying your text or check your API key.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("If you're seeing API errors, please check your SerpAPI key and usage limits.")

# Sidebar information
st.sidebar.title("ℹ️ About")
st.sidebar.write("""
### How it works:
1. Text Analysis: Your input is processed using:
   - Sentence-based semantic analysis (70%)
   - TF-IDF vectorization (30%)
2. Comprehensive Search: 
   - Checks maximum available sources
   - Analyzes complete sentences
3. Enhanced Detection:
   - Sentence-level matching
   - Progressive similarity scoring
   - Smart coverage boosting
4. Results: 
   - Sentence-based coverage
   - Internet content analysis
   - Detailed source matching
   - Comprehensive reporting

### Features:
- Sentence-level analysis
- Enhanced internet content detection
- Maximum source checking
- Smart similarity boosting
- Detailed reporting
- Weighted sentence matching

### Matching Levels:
- Strong Match: >70% similarity
- Moderate Match: >50% similarity
- Weak Match: >40% similarity
""")

# Add API status indicator
st.sidebar.write("---")
if api_key:
    st.sidebar.success("✅ SerpAPI: Configured")
else:
    st.sidebar.error("❌ SerpAPI: Not configured")