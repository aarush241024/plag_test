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
    sentence_scores = defaultdict(lambda: {'exact': 0, 'semantic': 0})
    sources = defaultdict(list)
    
    total_sources = len(search_results)
    for idx, (title, snippet, link) in enumerate(search_results):
        try:
            progress = (idx + 1) / total_sources
            st.progress(progress, text=f"Analyzing source {idx + 1} of {total_sources}")
            
            source_text = clean_text(f"{title} {snippet}")
            source_sentences = split_into_sentences(source_text)
            
            if not source_sentences:
                continue
                
            for i, input_sentence in enumerate(input_sentences):
                for source_sentence in source_sentences:
                    exact_score = calculate_exact_match_score(input_sentence, source_sentence)
                    semantic_score = calculate_semantic_similarity(input_sentence, source_sentence)
                    
                    if exact_score > sentence_scores[i]['exact']:
                        sentence_scores[i]['exact'] = exact_score
                    if semantic_score > sentence_scores[i]['semantic']:
                        sentence_scores[i]['semantic'] = semantic_score
                    
                    if exact_score >= similarity_threshold:
                        match_type = (
                            'exact' if exact_score >= 90 else
                            'similar' if exact_score >= 70 else
                            'paraphrase' if semantic_score >= 80 and exact_score < 70 else
                            'low_similarity'
                        )
                        
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
                        sources[link].append(match_data)
                        
        except Exception as e:
            continue
    
    total_sentences = len(input_sentences)
    if total_sentences > 0:
        all_scores = [scores['exact'] for scores in sentence_scores.values()]
        
        exact_matches = sum(1 for score in all_scores if score >= 90)
        similar_matches = sum(1 for score in all_scores if 70 <= score < 90)
        paraphrase_matches = sum(1 for i in range(total_sentences)
                               if sentence_scores[i]['semantic'] >= 80 
                               and sentence_scores[i]['exact'] < 70)
        
        exact_match_percent = (exact_matches / total_sentences) * 100
        similar_content_percent = (similar_matches / total_sentences) * 100
        paraphrase_percent = (paraphrase_matches / total_sentences) * 100
        
        # Calculate internet content score
        matched_sentences = sum(1 for score in all_scores if score >= similarity_threshold)
        internet_content_percent = (matched_sentences / total_sentences) * 100
    else:
        exact_match_percent = similar_content_percent = paraphrase_percent = internet_content_percent = 0
    
    return sentence_results, exact_match_percent, similar_content_percent, internet_content_percent, paraphrase_percent, input_sentences, sources

def generate_report(results, exact_percent, similar_percent, internet_percent, 
                   paraphrase_percent, similarity_threshold, sources_analyzed, sources):
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

    for source_link, matches in sources.items():
        source_title = matches[0]['source_title']
        max_score = max(m['exact_score'] for m in matches)
        avg_score = sum(m['exact_score'] for m in matches) / len(matches)
        
        report += f"""
Source: {source_title}
URL: {source_link}
Max Match Score: {max_score:.1f}%
Average Score: {avg_score:.1f}%
Total Matches: {len(matches)}

Matches:"""

        for match in sorted(matches, key=lambda x: x['exact_score'], reverse=True):
            report += f"""
- Match Score: {match['exact_score']:.1f}%
  Input Text: {match['input_sentence']}
  Matching Text: {match['matching_text']}
  Semantic Score: {match['semantic_score']:.1f}%
"""
    
    return report

def main():
    st.title("📝 Advanced Plagiarism Detector")
    st.markdown("*With Enhanced Internet Content Detection*")
    
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
                        internet_content_percent, paraphrase_percent, input_sentences, sources = \
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
                        
                        if results:
                            st.subheader("📚 Source Analysis")
                            
                            for source_link, matches in sorted(
                                sources.items(),
                                key=lambda x: max(m['exact_score'] for m in x[1]),
                                reverse=True
                            ):
                                source_title = matches[0]['source_title']
                                max_score = max(m['exact_score'] for m in matches)
                                
                                with st.expander(f"🔍 {source_title} - Max Match: {max_score:.1f}%"):
                                    st.markdown(f"**Source URL:** [{source_link}]({source_link})")
                                    st.markdown(f"**Number of Matches:** {len(matches)}")
                                    
                                    st.markdown("### Matching Content")
                                    for match in sorted(matches, key=lambda x: x['exact_score'], reverse=True):
                                        with st.container():
                                            score_color = (
                                                "🔴" if match['exact_score'] >= 90 else
                                                "🟡" if match['exact_score'] >= 70 else
                                                "🔵"
                                            )
                                            
                                            st.markdown(f"""
                                            {score_color} **Match Score: {match['exact_score']:.1f}%**
                                            
                                            **Your Text:**
                                            {match['input_sentence']}
                                            
                                            **Matched Text:**
                                            {match['matching_text']}
                                            
                                            **Match Details:**
                                            - Exact Score: {match['exact_score']:.1f}%
                                            - Semantic Score: {match['semantic_score']:.1f}%
                                            - Type: {match['match_type'].title()}
                                            """)
                                            st.markdown("---")
                            
                            match_level = "High" if internet_content_percent > 80 else "Moderate" if internet_content_percent > 50 else "Low"
                            st.warning(f"⚠️ {match_level} level of internet content detected!")
                            
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
                                    sources
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
