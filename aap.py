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
from collections import defaultdict

load_dotenv()

DEFAULT_SIMILARITY_THRESHOLD = 60
DEFAULT_SEARCH_DEPTH = 50

EXACT_MATCH_THRESHOLD = 85
HIGH_SIMILARITY_THRESHOLD = 70
SEMANTIC_MATCH_THRESHOLD = 75
INTERNET_CONTENT_HIGH = 80
INTERNET_CONTENT_MODERATE = 50

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
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'[^\w\s.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text.strip()

def split_into_sentences(text):
    try:
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    except Exception:
        sentences = [s.strip() + '.' for s in text.split('.') if len(s.strip()) > 20]
        return sentences

def calculate_exact_match_score(text1, text2):
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    if text1 == text2:
        return 100.0
    
    char_similarity = SequenceMatcher(None, text1, text2).ratio() * 100
    
    words1 = text1.split()
    words2 = text2.split()
    
    common_words = set(words1) & set(words2)
    unique_words = set(words1) | set(words2)
    
    word_similarity = (len(common_words) / len(unique_words) * 100) if unique_words else 0
    
    words1_str = ' '.join(words1)
    words2_str = ' '.join(words2)
    
    matcher = SequenceMatcher(None, words1_str, words2_str)
    match = matcher.find_longest_match(0, len(words1_str), 0, len(words2_str))
    consecutive_score = (match.size / len(words1_str)) * 100 if words1_str else 0
    
    exact_match_score = (
        char_similarity * 0.3 +
        word_similarity * 0.3 +
        consecutive_score * 0.4
    )
    
    return exact_match_score

def calculate_semantic_similarity(text1, text2):
    try:
        embeddings = models['semantic'].encode([text1, text2], convert_to_tensor=True)
        embedding1 = embeddings[0].cpu().numpy()
        embedding2 = embeddings[1].cpu().numpy()
        similarity = np.dot(embedding1, embedding2)
        return similarity * 100
    except:
        return 0

def analyze_text_similarity(input_text, search_results, similarity_threshold):
    cleaned_input = clean_text(input_text)
    input_sentences = split_into_sentences(cleaned_input)
    
    if not input_sentences:
        return [], 0, 0, 0, 0, [], {}
    
    sentence_results = []
    sentence_scores = {}
    resource_matches = {}
    
    total_sources = len(search_results)
    for idx, (title, snippet, link) in enumerate(search_results):
        try:
            progress = (idx + 1) / total_sources
            st.progress(progress, text=f"Analyzing source {idx + 1} of {total_sources}")
            
            source_text = clean_text(f"{title} {snippet}")
            source_sentences = split_into_sentences(source_text)
            current_resource_matches = []
            
            if not source_sentences:
                continue
            
            for i, input_sentence in enumerate(input_sentences):
                for source_sentence in source_sentences:
                    exact_score = calculate_exact_match_score(input_sentence, source_sentence)
                    semantic_score = calculate_semantic_similarity(input_sentence, source_sentence)
                    
                    current_best = sentence_scores.get(i, {'exact': 0, 'semantic': 0})
                    if exact_score > current_best['exact']:
                        current_best['exact'] = exact_score
                    if semantic_score > current_best['semantic']:
                        current_best['semantic'] = semantic_score
                    sentence_scores[i] = current_best
                    
                    if exact_score >= similarity_threshold:
                        if exact_score >= 90:
                            match_type = 'exact'
                        elif exact_score >= 70:
                            match_type = 'similar'
                        elif semantic_score >= 80 and exact_score < 70:
                            match_type = 'paraphrase'
                        else:
                            match_type = 'low_similarity'
                        
                        match_data = {
                            'sentence_index': i,
                            'input_sentence': input_sentence,
                            'matching_text': source_sentence,
                            'exact_score': exact_score,
                            'semantic_score': semantic_score,
                            'match_type': match_type,
                            'source_title': title,
                            'source_link': link
                        }
                        
                        sentence_results.append(match_data)
                        current_resource_matches.append(match_data)
            
            if current_resource_matches:
                resource_matches[link] = {
                    'title': title,
                    'link': link,
                    'matches': current_resource_matches,
                    'match_count': len(current_resource_matches),
                    'avg_score': sum(m['exact_score'] for m in current_resource_matches) / len(current_resource_matches),
                    'max_score': max(m['exact_score'] for m in current_resource_matches),
                    'types': {
                        'exact': sum(1 for m in current_resource_matches if m['match_type'] == 'exact'),
                        'similar': sum(1 for m in current_resource_matches if m['match_type'] == 'similar'),
                        'paraphrase': sum(1 for m in current_resource_matches if m['match_type'] == 'paraphrase')
                    }
                }
                
        except Exception as e:
            continue
    
    total_sentences = len(input_sentences)
    if total_sentences > 0:
        exact_matches = sum(1 for i in range(total_sentences) if sentence_scores.get(i, {'exact': 0})['exact'] >= 90)
        similar_matches = sum(1 for i in range(total_sentences) if 70 <= sentence_scores.get(i, {'exact': 0})['exact'] < 90)
        semantic_matches = sum(1 for i in range(total_sentences) if sentence_scores.get(i, {'semantic': 0})['semantic'] >= 80 and sentence_scores.get(i, {'exact': 0})['exact'] < 70)
        
        exact_match_percent = (exact_matches / total_sentences) * 100
        similar_content_percent = (similar_matches / total_sentences) * 100
        paraphrase_percent = (semantic_matches / total_sentences) * 100
        
        internet_score = 0
        for i in range(total_sentences):
            scores = sentence_scores.get(i, {'exact': 0, 'semantic': 0})
            exact_score = scores['exact']
            semantic_score = scores['semantic']
            
            if exact_score >= 90:
                internet_score += 1.0
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
        
        base_internet_percent = (internet_score / total_sentences) * 100
        
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
    
    return (sentence_results, exact_match_percent, similar_content_percent, internet_content_percent, paraphrase_percent, input_sentences, resource_matches)

def generate_report(results, exact_percent, similar_percent, internet_percent, paraphrase_percent, similarity_threshold, sources_analyzed, resource_matches):
    report = f"""Plagiarism Analysis Report

Overall Metrics:
---------------
Exact Matches: {exact_percent:.1f}%
Similar Content: {similar_percent:.1f}%
Internet Content: {internet_percent:.1f}%
Potential Paraphrasing: {paraphrase_percent:.1f}%
Similarity Threshold: {similarity_threshold}%
Sources Analyzed: {sources_analyzed}

Source Analysis:
--------------"""

    for resource in sorted(resource_matches.values(), key=lambda x: x['max_score'], reverse=True):
        report += f"""
Source: {resource['title']}
URL: {resource['link']}
- Total Matches: {resource['match_count']}
- Average Score: {resource['avg_score']:.1f}%
- Maximum Score: {resource['max_score']:.1f}%
- Match Types:
  * Exact: {resource['types']['exact']}
  * Similar: {resource['types']['similar']}
  * Paraphrase: {resource['types']['paraphrase']}

Matches:"""
        
        for match in sorted(resource['matches'], key=lambda x: x['exact_score'], reverse=True):
            report += f"""
- Match Score: {match['exact_score']:.1f}%
  Input Text: {match['input_sentence']}
  Matching Text: {match['matching_text']}
  Type: {match['match_type'].upper()}
"""
    
    return report

def main():
    st.title("📝 Advanced Plagiarism Detector")
    st.markdown("*With Enhanced Source Analysis*")
    
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
    
    input_text = st.text_area("Input Text", height=200)
    
    st.sidebar.title("Settings")
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold (%)", 
        min_value=0, 
        max_value=100, 
        value=DEFAULT_SIMILARITY_THRESHOLD,
        help="Adjust the minimum similarity threshold for match detection"
    )
    
    search_depth = st.sidebar.slider(
        "Search Depth", 
        min_value=10, 
        max_value=100, 
        value=DEFAULT_SEARCH_DEPTH,
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
                        internet_content_percent, paraphrase_percent, input_sentences, resource_matches = \
                        analyze_text_similarity(input_text, search_results, similarity_threshold)
                        
                        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                        with col1:
                            st.metric("Exact Matches", f"{exact_match_percent:.1f}%")
                            st.metric("Average Score", f"{sum(r['exact_score'] for r in results) / len(results):.1f}%" if results else "0.0%")
                        with col2:
                            st.metric("Similar Content", f"{similar_content_percent:.1f}%")
                            st.metric("Max Score", f"{max((r['exact_score'] for r in results), default=0):.1f}%")
                        with col3:
                            internet_label = "Internet Content 🌐"
                            if internet_content_percent > 80:
                                st.metric(internet_label, f"{internet_content_percent:.1f}%", delta="High", delta_color="inverse")
                            elif internet_content_percent > 50:
                                st.metric(internet_label, f"{internet_content_percent:.1f}%", delta="Medium", delta_color="off")
                            else:
                                st.metric(internet_label, f"{internet_content_percent:.1f}%", delta="Low")
                            st.metric("Matched Sentences", f"{len(set(r['sentence_index'] for r in results))}/{len(input_sentences)}" if results else "0/0")
                        with col4:
                            st.metric("Possible Paraphrasing", f"{paraphrase_percent:.1f}%")
                            st.write("Score Details...")

                        
                        if results:
                            st.subheader("📚 Source Analysis")
                            
                            sorted_resources = sorted(
                                resource_matches.values(),
                                key=lambda x: (x['match_count'], x['max_score']),
                                reverse=True
                            )
                            
                            for resource in sorted_resources:
                                with st.expander(
                                    f"🔍 {resource['title']} "
                                    f"({resource['match_count']} matches, "
                                    f"Max Score: {resource['max_score']:.1f}%)"
                                ):
                                    st.markdown(f"""
                                    **Source Details**
                                    - URL: [{resource['link']}]({resource['link']})
                                    - Total Matches: {resource['match_count']}
                                    - Average Match Score: {resource['avg_score']:.1f}%
                                    - Maximum Match Score: {resource['max_score']:.1f}%
                                    
                                    **Match Types**
                                    - Exact Matches: {resource['types']['exact']}
                                    - Similar Content: {resource['types']['similar']}
                                    - Potential Paraphrases: {resource['types']['paraphrase']}
                                    """)
                                    
                                    st.markdown("### Matching Content")
                                    for match in sorted(
                                        resource['matches'],
                                        key=lambda x: x['exact_score'],
                                        reverse=True
                                    ):
                                        with st.container():
                                            if match['match_type'] == 'exact':
                                                st.error(f"⚠️ Exact Match ({match['exact_score']:.1f}%)")
                                            elif match['match_type'] == 'similar':
                                                st.warning(f"⚡ Similar Content ({match['exact_score']:.1f}%)")
                                            else:
                                                st.info(f"🔄 Potential Paraphrase ({match['exact_score']:.1f}%)")
                                                
                                            st.markdown(f"""
                                            **Your Text:**
                                            {match['input_sentence']}
                                            
                                            **Matching Text:**
                                            {match['matching_text']}
                                            
                                            **Similarity Scores:**
                                            - Match Score: {match['exact_score']:.1f}%
                                            - Semantic Score: {match['semantic_score']:.1f}%
                                            """)
                                            st.markdown("---")
                            
                            match_level = "High" if internet_content_percent > 80 else "Moderate" if internet_content_percent > 50 else "Low"
                            st.warning(f"⚠️ {match_level} level of internet content detected!")
                            
                            # Source analysis summary
                            st.info(f"""
                            📊 Analysis Summary:
                            - Total Sources: {len(sources)}
                            - Total Matches: {len(results)}
                            - Matched Sentences: {len(set(r['sentence_index'] for r in results))}/{len(input_sentences)}
                            - Overall Similarity: {internet_content_percent:.1f}%
                            - Average Match Score: {sum(r['exact_score'] for r in results) / len(results):.1f}%
                            - Maximum Match Score: {max(r['exact_score'] for r in results):.1f}%
                            
                            Match Level: {match_level.upper()}
                            """)
                            
                            if st.button("📥 Download Detailed Report"):
                                report = generate_report(
                                    results,
                                    exact_match_percent,
                                    similar_content_percent,
                                    internet_content_percent,
                                    paraphrase_percent,
                                    similarity_threshold,
                                    len(search_results),
                                    resource_matches
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