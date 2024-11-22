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
from difflib import SequenceMatcher
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Load environment variables
load_dotenv()

# Initialize the models
@st.cache_resource
def load_models():
    return {
        'semantic': SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),
        'tfidf': TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words='english'
        )
    }

models = load_models()

def get_search_results(query, num_results=100):
    """Get search results using SerpAPI"""
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERPAPI_KEY"),
            "engine": "google",
            "num": num_results,
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
    """Clean and normalize text for comparison"""
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    # Normalize whitespace and punctuation
    text = re.sub(r'[^\w\s.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Normalize case
    text = text.lower()
    return text.strip()

def split_into_sentences(text):
    """Split text into sentences with improved handling"""
    try:
        # Normalize periods and maintain sentence boundaries
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)
        sentences = sent_tokenize(text)
        # Filter out very short sentences
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    except Exception:
        # Fallback to simple splitting
        sentences = [s.strip() + '.' for s in text.split('.') if len(s.strip()) > 20]
        return sentences

def calculate_exact_match_score(text1, text2):
    """Calculate exact match score"""
    # Clean and normalize texts
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    # Direct string comparison for exact matches
    if text1 == text2:
        return 100.0
        
    # Calculate character-level similarity
    char_similarity = SequenceMatcher(None, text1, text2).ratio() * 100
    
    # Calculate word-level matches
    words1 = text1.split()
    words2 = text2.split()
    
    # Word sets for overlap calculation
    common_words = set(words1) & set(words2)
    unique_words = set(words1) | set(words2)
    
    # Word overlap percentage
    word_similarity = (len(common_words) / len(unique_words) * 100) if unique_words else 0
    
    # Calculate consecutive word matches
    words1_str = ' '.join(words1)
    words2_str = ' '.join(words2)
    
    # Find longest common substring
    matcher = SequenceMatcher(None, words1_str, words2_str)
    match = matcher.find_longest_match(0, len(words1_str), 0, len(words2_str))
    consecutive_score = (match.size / len(words1_str)) * 100 if words1_str else 0
    
    # Combined score weighted towards exact matches
    exact_match_score = (
        char_similarity * 0.3 +     # Character-level similarity
        word_similarity * 0.3 +     # Word overlap
        consecutive_score * 0.4     # Consecutive word matches
    )
    
    return exact_match_score

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity for paraphrase detection"""
    try:
        # Get embeddings
        embeddings = models['semantic'].encode([text1, text2], convert_to_tensor=True)
        
        # Convert to numpy for calculation
        embedding1 = embeddings[0].cpu().numpy()
        embedding2 = embeddings[1].cpu().numpy()
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return similarity * 100
    except:
        return 0

def analyze_text_similarity(input_text, search_results, similarity_threshold):
    """Analyze text for exact matches, similar content, and paraphrasing"""
    cleaned_input = clean_text(input_text)
    input_sentences = split_into_sentences(cleaned_input)
    
    if not input_sentences:
        return [], 0, 0, 0, 0, []
    
    # Results storage
    sentence_results = []
    sentence_scores = {}  # Store best scores for each sentence
    
    total_sources = len(search_results)
    for idx, (title, snippet, link) in enumerate(search_results):
        try:
            progress = (idx + 1) / total_sources
            st.progress(progress, text=f"Analyzing source {idx + 1} of {total_sources}")
            
            source_text = clean_text(f"{title} {snippet}")
            source_sentences = split_into_sentences(source_text)
            
            if not source_sentences:
                continue
            
            # Check each input sentence
            for i, input_sentence in enumerate(input_sentences):
                # Check against each source sentence
                for source_sentence in source_sentences:
                    # Calculate both similarity scores
                    exact_score = calculate_exact_match_score(input_sentence, source_sentence)
                    semantic_score = calculate_semantic_similarity(input_sentence, source_sentence)
                    
                    # Update best scores for this sentence
                    current_best = sentence_scores.get(i, {'exact': 0, 'semantic': 0})
                    if exact_score > current_best['exact']:
                        current_best['exact'] = exact_score
                    if semantic_score > current_best['semantic']:
                        current_best['semantic'] = semantic_score
                    sentence_scores[i] = current_best
                    
                    # Only process if above threshold
                    if exact_score >= similarity_threshold:
                        # Determine match type based on scores
                        if exact_score >= 90:
                            match_type = 'exact'
                        elif exact_score >= 70:
                            match_type = 'similar'
                        elif semantic_score >= 80 and exact_score < 70:
                            match_type = 'paraphrase'
                        else:
                            match_type = 'low_similarity'
                        
                        sentence_results.append({
                            'sentence_index': i,
                            'input_sentence': input_sentence,
                            'matching_text': source_sentence,
                            'exact_score': exact_score,
                            'semantic_score': semantic_score,
                            'match_type': match_type,
                            'source_title': title,
                            'source_link': link
                        })
        except Exception as e:
            continue
    
    # Calculate percentages based on threshold
    total_sentences = len(input_sentences)
    if total_sentences > 0:
        # Calculate match percentages
        qualifying_matches = [r for r in sentence_results if r['exact_score'] >= similarity_threshold]
        
        # Count different types of matches using best scores
        exact_matches = sum(1 for i in range(total_sentences) 
                          if sentence_scores.get(i, {'exact': 0})['exact'] >= 90)
        similar_matches = sum(1 for i in range(total_sentences)
                            if 70 <= sentence_scores.get(i, {'exact': 0})['exact'] < 90)
        semantic_matches = sum(1 for i in range(total_sentences)
                             if sentence_scores.get(i, {'semantic': 0})['semantic'] >= 80
                             and sentence_scores.get(i, {'exact': 0})['exact'] < 70)
        
        # Calculate individual percentages
        exact_match_percent = (exact_matches / total_sentences) * 100
        similar_content_percent = (similar_matches / total_sentences) * 100
        paraphrase_percent = (semantic_matches / total_sentences) * 100
        
        # Enhanced internet content calculation with progressive scoring
        internet_score = 0
        for i in range(total_sentences):
            scores = sentence_scores.get(i, {'exact': 0, 'semantic': 0})
            exact_score = scores['exact']
            semantic_score = scores['semantic']
            
            # Progressive scoring based on match quality
            if exact_score >= 90:
                internet_score += 1.0  # Full score for exact matches
            elif exact_score >= 80:
                internet_score += 0.9
            elif exact_score >= 70:
                internet_score += 0.8
            elif exact_score >= similarity_threshold:
                internet_score += 0.7
            elif semantic_score >= 80:
                internet_score += 0.6
            elif semantic_score >= 70:
                internet_score += 0.4
            elif semantic_score >= 60:
                internet_score += 0.2
        
        # Calculate base internet percentage
        base_internet_percent = (internet_score / total_sentences) * 100
        
        # Apply boost based on overall match quality
        boost = 1.0
        match_ratio = (exact_matches + similar_matches) / total_sentences
        if match_ratio > 0.8:
            boost = 1.3
        elif match_ratio > 0.6:
            boost = 1.2
        elif match_ratio > 0.4:
            boost = 1.1
        
        internet_content_percent = min(base_internet_percent * boost, 100)
        
    else:
        exact_match_percent = similar_content_percent = paraphrase_percent = internet_content_percent = 0
    
    # Filter and sort results
    filtered_results = []
    seen_sentences = set()
    
    # Sort by exact_score and keep highest scoring match for each sentence
    sorted_results = sorted(sentence_results, key=lambda x: x['exact_score'], reverse=True)
    for result in sorted_results:
        if result['exact_score'] >= similarity_threshold and result['sentence_index'] not in seen_sentences:
            filtered_results.append(result)
            seen_sentences.add(result['sentence_index'])
    
    return (filtered_results, exact_match_percent, similar_content_percent, 
            internet_content_percent, paraphrase_percent, input_sentences)

def generate_report(results, exact_percent, similar_percent, internet_percent, 
                   paraphrase_percent, threshold, sources_analyzed):
    """Generate detailed analysis report"""
    report = f"""Plagiarism Analysis Report
    
    Overall Metrics:
    ---------------
    Exact Matches: {exact_percent:.1f}%
    Similar Content: {similar_percent:.1f}%
    Internet Content: {internet_percent:.1f}%
    Potential Paraphrasing: {paraphrase_percent:.1f}%
    Similarity Threshold: {threshold}%
    Sources Analyzed: {sources_analyzed}
    
    Detailed Analysis:
    ----------------"""
    
    for idx, result in enumerate(results, 1):
        report += f"""
        Match #{idx}:
        - Input Text: {result['input_sentence']}
        - Match Type: {result['match_type'].upper()}
        - Exact Match Score: {result['exact_score']:.1f}%
        - Semantic Similarity: {result['semantic_score']:.1f}%
        - Matching Text: {result['matching_text']}
        - Source: {result['source_link']}
        
        """
    
    return report

def main():
    st.title("📝 Advanced Plagiarism Detector")
    st.markdown("*With Enhanced Internet Content Detection*")
    
    # API Status Check
    try:
        api_key = os.getenv('SERPAPI_KEY')
        if not api_key:
            st.error("⚠️ SERPAPI_KEY not found in .env file. Please configure your API key.")
            st.sidebar.error("❌ SerpAPI: Not configured")
            st.stop()
        else:
            st.sidebar.success("✅ SerpAPI: Configured")
    except Exception as e:
        st.error(f"Error accessing API key: {str(e)}")
        st.stop()
    
    # Text input
    input_text = st.text_area("Input Text", height=200)
    
    # Advanced settings in sidebar
    st.sidebar.title("Settings")
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold (%)", 
        min_value=0, 
        max_value=100, 
        value=60,
        help="Adjust the minimum similarity threshold for match detection"
    )
    
    search_depth = st.sidebar.slider(
        "Search Depth", 
        min_value=10, 
        max_value=100, 
        value=50,
        help="Number of search results to analyze"
    )
    
    if st.button("🔍 Check Plagiarism"):
        if len(input_text) < 50:
            st.error("Please enter at least 50 characters for accurate analysis.")
        else:
            with st.spinner("🔄 Analyzing text..."):
                try:
                    st.info("🌐 Searching for similar content...")
                    search_results = get_search_results(input_text, search_depth)
                    
                    if search_results:
                        st.info(f"Found {len(search_results)} sources to analyze...")
                        results, exact_match_percent, similar_content_percent, \
                        internet_content_percent, paraphrase_percent, input_sentences = \
                        analyze_text_similarity(input_text, search_results, similarity_threshold)
                        
                        # Display metrics with enhanced visibility
                        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                        with col1:
                            st.metric("Exact Matches", f"{exact_match_percent:.1f}%")
                        with col2:
                            st.metric("Similar Content", f"{similar_content_percent:.1f}%")
                        with col3:
                            internet_label = "Internet Content 🌐"
                            if internet_content_percent > 80:
                                st.metric(internet_label, f"{internet_content_percent:.1f}%", delta="High", delta_color="inverse")
                            elif internet_content_percent > 50:
                                st.metric(internet_label, f"{internet_content_percent:.1f}%", delta="Medium", delta_color="off")
                            else:
                                st.metric(internet_label, f"{internet_content_percent:.1f}%", delta="Low")
                        with col4:
                            st.metric("Possible Paraphrasing", f"{paraphrase_percent:.1f}%")
                        
                        if results:
                            match_level = "High" if internet_content_percent > 80 else "Moderate" if internet_content_percent > 50 else "Low"
                            st.warning(f"⚠️ {match_level} level of internet content detected!")
                            
                            # Analysis summary with enhanced internet content reporting
                            st.info(f"""
                            📊 Analysis Summary:
                            - {internet_content_percent:.1f}% of content appears to be from internet sources
                            - {exact_match_percent:.1f}% exact matches found
                            - {similar_content_percent:.1f}% similar content detected
                            - {paraphrase_percent:.1f}% potentially paraphrased content
                            - Analyzed {len(search_results)} potential sources
                            - Current similarity threshold: {similarity_threshold}%
                            
                            Internet Content Level: {match_level}
                            {'🔴 High probability of internet-sourced content' if match_level == 'High'
                             else '🟡 Moderate presence of internet content' if match_level == 'Moderate'
                             else '🟢 Low indication of internet content'}
                            """)
                            
                            # Display detailed results
                            for result in results:
                                match_color = {
                                    'exact': 'red',
                                    'similar': 'orange',
                                    'paraphrase': 'blue'
                                }.get(result['match_type'], 'grey')
                                
                                with st.expander(
                                    f"Sentence #{result['sentence_index'] + 1} - "
                                    f"{result['exact_score']:.1f}% Match"
                                ):
                                    st.markdown("**Your Sentence:**")
                                    st.write(result['input_sentence'])
                                    st.markdown("**Matching Source:**")
                                    st.write(result['matching_text'])
                                    st.markdown(f"**Source:** [{result['source_link']}]({result['source_link']})")
                                    
                                    # Enhanced match type indicators
                                    if result['match_type'] == 'exact':
                                        st.error("⚠️ Exact internet match detected!")
                                    elif result['match_type'] == 'similar':
                                        st.warning("⚡ High similarity to internet content")
                                    elif result['match_type'] == 'paraphrase':
                                        st.info("🔄 Potential paraphrase of internet content")
                                    
                                    st.markdown(f"""
                                    **Match Details:**
                                    - Exact Match Score: {result['exact_score']:.1f}%
                                    - Semantic Similarity: {result['semantic_score']:.1f}%
                                    - Match Type: {result['match_type'].title()}
                                    - Content Source: Internet
                                    """)
                            
                            # Generate report
                            if st.button("📥 Download Detailed Report"):
                                report = generate_report(
                                    results, exact_match_percent, similar_content_percent,
                                    internet_content_percent, paraphrase_percent,
                                    similarity_threshold, len(search_results)
                                )
                                
                                st.download_button(
                                    "📄 Save Report",
                                    report,
                                    "plagiarism_report.txt",
                                    "text/plain"
                                )
                        else:
                            st.success(f"✅ No significant matches found above {similarity_threshold}% similarity threshold!")
                    else:
                        st.warning("No search results found. Try modifying your text or check your API key.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("If you're seeing API errors, please check your SerpAPI key and usage limits.")

    # Sidebar information
    st.sidebar.title("ℹ️ About")
    st.sidebar.write(f"""
    ### Analysis Types:
    1. Internet Content:
       - Overall presence in online sources
       - Weighted by match quality
       - Progressive scoring system

    2. Match Categories:
       - Exact Match: ≥90% similarity
       - High Similarity: 70-90%
       - Paraphrase: High semantic match

    3. Scoring System:
       - Direct Copies: 100%
       - Near Matches: 70-90%
       - Paraphrased: Based on meaning

    ### Current Settings:
    - Similarity Threshold: {similarity_threshold}%
    - Search Depth: {search_depth} sources
    """)

if __name__ == "__main__":
    main()