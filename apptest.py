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
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

load_dotenv()

DEFAULT_SIMILARITY_THRESHOLD = 10
DEFAULT_SEARCH_DEPTH = 50
MIN_QUERY_TERMS = 3
MAX_QUERY_TERMS = 10

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

def preprocess_text(text):
    """Preprocess text for vectorization"""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def get_text_stats(text):
    """Get word count and character count for a text"""
    words = word_tokenize(clean_text(text))
    return {
        'word_count': len(words),
        'char_count': len(text)
    }

def calculate_text_ratio(input_text, source_text):
    """Calculate the ratio between input text and source text"""
    input_stats = get_text_stats(input_text)
    source_stats = get_text_stats(source_text)
    
    word_ratio = (input_stats['word_count'] / source_stats['word_count'] 
                 if source_stats['word_count'] > 0 else 0)
    char_ratio = (input_stats['char_count'] / source_stats['char_count'] 
                 if source_stats['char_count'] > 0 else 0)
    
    return (word_ratio + char_ratio) / 2 * 100

def extract_key_terms(text, n_terms=8):
    """Extract key terms from text using TF-IDF"""
    processed_text = preprocess_text(text)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_text])
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    
    term_scores = [(term, score) for term, score in zip(feature_names, scores)]
    term_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [term for term, _ in term_scores[:n_terms]]

def generate_search_queries(text):
    """Generate multiple search queries from input text"""
    sentences = sent_tokenize(text)
    queries = []
    
    key_terms = extract_key_terms(text)
    if key_terms:
        queries.append(' '.join(key_terms[:MAX_QUERY_TERMS]))
    
    for sentence in sentences:
        terms = extract_key_terms(sentence, n_terms=MAX_QUERY_TERMS)
        if len(terms) >= MIN_QUERY_TERMS:
            queries.append(' '.join(terms))
    
    return list(set(queries))[:5]

def get_search_results(text, num_results=100):
    """Enhanced search with multiple queries using SerpAPI"""
    all_results = []
    seen_links = set()
    
    queries = generate_search_queries(text)
    
    for query in queries:
        try:
            search = GoogleSearch({
                "q": query,
                "api_key": os.getenv("SERPAPI_KEY"),
                "engine": "google",
                "num": num_results // len(queries),
                "gl": "us",
                "hl": "en"
            })
            
            results = search.get_dict()
            
            if "error" in results:
                st.error(f"SerpAPI Error: {results['error']}")
                continue
                
            if "organic_results" in results:
                for result in results["organic_results"]:
                    link = result.get("link", "")
                    if link not in seen_links:
                        seen_links.add(link)
                        all_results.append((
                            result.get("title", ""),
                            result.get("snippet", ""),
                            link
                        ))
                        
        except Exception as e:
            st.error(f"Search API Error: {str(e)}")
            continue
            
    return all_results[:num_results]

def clean_text(text):
    """Clean and normalize text"""
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'[^\w\s.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text.strip()

def calculate_exact_match_score(text1, text2):
    """Calculate similarity score between two texts"""
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    if text1 == text2:
        return 100.0

    def get_ngrams(text, n):
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    words1 = text1.split()
    words2 = text2.split()
    
    unigrams1 = set(words1)
    unigrams2 = set(words2)
    bigrams1 = set(get_ngrams(text1, 2))
    bigrams2 = set(get_ngrams(text2, 2))
    trigrams1 = set(get_ngrams(text1, 3))
    trigrams2 = set(get_ngrams(text2, 3))
    
    def get_common_sequences(seq1, seq2, min_length=3):
        sequences = []
        len1, len2 = len(seq1), len(seq2)
        for i in range(len1):
            for j in range(len2):
                k = 0
                while (i + k < len1 and j + k < len2 and 
                       seq1[i + k] == seq2[j + k]):
                    k += 1
                if k >= min_length:
                    sequences.append(' '.join(seq1[i:i+k]))
        return sequences
    
    common_sequences = get_common_sequences(words1, words2)
    
    unigram_overlap = len(unigrams1 & unigrams2) / len(unigrams1 | unigrams2) * 100
    bigram_overlap = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2) * 100 if bigrams1 and bigrams2 else 0
    trigram_overlap = len(trigrams1 & trigrams2) / len(trigrams1 | trigrams2) * 100 if trigrams1 and trigrams2 else 0
    
    sequence_words = set(' '.join(common_sequences).split())
    sequence_coverage = len(sequence_words) / max(len(unigrams1), len(unigrams2)) * 100
    
    total_words = len(words1) + len(words2)
    sequence_length = sum(len(seq.split()) for seq in common_sequences)
    sequence_ratio = (sequence_length / total_words) * 200
    
    exact_match_score = (
        unigram_overlap * 0.2 +
        bigram_overlap * 0.2 +
        trigram_overlap * 0.2 +
        sequence_coverage * 0.2 +
        sequence_ratio * 0.2
    )
    
    return min(exact_match_score, 100.0)

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts"""
    try:
        embeddings = models['semantic'].encode([text1, text2], convert_to_tensor=True)
        embedding1 = embeddings[0].cpu().numpy()
        embedding2 = embeddings[1].cpu().numpy()
        similarity = np.dot(embedding1, embedding2)
        return similarity * 100
    except:
        return 0

def analyze_text_similarity(input_text, search_results, similarity_threshold):
    """Analyze text similarity with enhanced ratio-based matching"""
    cleaned_input = clean_text(input_text)
    results = []
    sources = defaultdict(list)
    
    input_stats = get_text_stats(cleaned_input)
    total_plagiarism_score = 0
    source_scores = []
    
    input_vector = models['semantic'].encode(cleaned_input, convert_to_tensor=True)
    
    total_sources = len(search_results)
    for idx, (title, snippet, link) in enumerate(search_results):
        try:
            progress = (idx + 1) / total_sources
            st.progress(progress, text=f"Analyzing source {idx + 1} of {total_sources}")
            
            source_text = clean_text(f"{title} {snippet}")
            if not source_text:
                continue
            
            exact_score = calculate_exact_match_score(cleaned_input, source_text)
            semantic_score = calculate_semantic_similarity(cleaned_input, source_text)
            text_ratio = calculate_text_ratio(cleaned_input, source_text)
            
            adjusted_exact_score = exact_score * text_ratio / 100
            
            if adjusted_exact_score >= similarity_threshold:
                match_type = 'exact' if adjusted_exact_score >= 90 else 'low_similarity'
                
                source_stats = get_text_stats(source_text)
                
                match_data = {
                    'input_text': cleaned_input,
                    'matching_text': source_text,
                    'exact_score': exact_score,
                    'adjusted_exact_score': adjusted_exact_score,
                    'semantic_score': semantic_score,
                    'text_ratio': text_ratio,
                    'match_type': match_type,
                    'source_title': title,
                    'source_link': link,
                    'source_word_count': source_stats['word_count'],
                    'input_word_count': input_stats['word_count']
                }
                
                results.append(match_data)
                sources[link].append(match_data)
                
                source_scores.append({
                    'score': adjusted_exact_score,
                    'word_count': source_stats['word_count']
                })
                        
        except Exception as e:
            continue

    if source_scores:
        total_words = sum(score['word_count'] for score in source_scores)
        total_plagiarism_score = sum(
            score['score'] * (score['word_count'] / total_words)
            for score in source_scores
        )
    
    if results:
        max_exact = max(r['adjusted_exact_score'] for r in results)
        exact_matches = sum(1 for r in results if r['adjusted_exact_score'] >= 90)
        
        exact_match_percent = max_exact
        internet_content_percent = total_plagiarism_score
    else:
        exact_match_percent = internet_content_percent = 0
    
    return (results, exact_match_percent, internet_content_percent, [cleaned_input], sources)

def generate_report(results, exact_percent, internet_percent, 
                   similarity_threshold, sources_analyzed, sources):
    """Generate a detailed analysis report with text ratio information"""
    report = f"""Plagiarism Analysis Report

Overall Metrics:
---------------
Final Plagiarism Score: {internet_percent:.1f}%
Exact Matches: {exact_percent:.1f}%
Similarity Threshold: {similarity_threshold}%
Sources Analyzed: {sources_analyzed}

Analysis Methodology:
-------------------
- Text ratio analysis comparing input and source lengths
- Semantic similarity detection using transformer models
- N-gram analysis for exact matching
- Weighted scoring based on text overlap
- Length-adjusted similarity calculations

Detailed Source Analysis:
-----------------------"""

    for source_link, matches in sources.items():
        source_title = matches[0]['source_title']
        max_score = max(m['adjusted_exact_score'] for m in matches)
        avg_score = sum(m['adjusted_exact_score'] for m in matches) / len(matches)
        max_ratio = max(m['text_ratio'] for m in matches)
        
        report += f"""

Source: {source_title}
URL: {source_link}
Max Match Score (Adjusted): {max_score:.1f}%
Average Score (Adjusted): {avg_score:.1f}%
Text Length Ratio: {max_ratio:.1f}%
Total Matches: {len(matches)}

Individual Matches:"""

        for match in sorted(matches, key=lambda x: x['adjusted_exact_score'], reverse=True):
            report += f"""
- Match Score: {match['adjusted_exact_score']:.1f}% (Original: {match['exact_score']:.1f}%)
  Text Ratio: {match['text_ratio']:.1f}%
  Input Words: {match['input_word_count']}
  Source Words: {match['source_word_count']}
  
  Input Text:
  {match['input_text']}
  
  Matching Text:
  {match['matching_text']}
  
  Additional Metrics:
  - Semantic Score: {match['semantic_score']:.1f}%
  - Match Type: {match['match_type']}
"""

    report += """

Recommendations:
--------------"""
    
    if internet_percent > INTERNET_CONTENT_HIGH:
        report += """
- High level of matched content detected
- Significant revision recommended for sections with high similarity
- Consider rewriting matched passages using original language
- Review all sources and add proper citations where needed"""
    elif internet_percent > INTERNET_CONTENT_MODERATE:
        report += """
- Moderate level of matched content found
- Some revision may be needed for highly similar sections
- Review matches and consider rephrasing
- Ensure all sources are properly cited"""
    else:
        report += """
- Low level of matched content detected
- Minor revisions may improve originality
- Consider reviewing any exact matches
- Ensure proper citation for any referenced material"""

    report += f"""

Analysis Summary:
---------------
- Total Sources Analyzed: {sources_analyzed}
- Sources with Matches: {len(sources)}
- Overall Similarity: {internet_percent:.1f}%

Note: Scores are adjusted based on the ratio of text lengths between input and sources.
"""
    
    return report

def main():
    st.title("📝 Enhanced Plagiarism Detector")
    st.markdown("*With Advanced Text Ratio Analysis*")
    
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
    
    input_text = st.text_area("Input Text for Analysis", height=200)
    
    st.sidebar.title("Analysis Settings")
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
    
    if st.button("🔍 Analyze Text"):
        if len(input_text) < 50:
            st.error("Please enter at least 50 characters for accurate analysis.")
        else:
            with st.spinner("🔄 Processing text..."):
                try:
                    st.info("📊 Vectorizing input text and extracting key terms...")
                    key_terms = extract_key_terms(input_text)
                    if key_terms:
                        st.success(f"Key terms identified: {', '.join(key_terms)}")
                    
                    st.info("🌐 Searching for similar content...")
                    search_results = get_search_results(input_text, search_depth)
                    
                    if search_results:
                        st.info(f"Found {len(search_results)} sources to analyze...")
                        
                        results, exact_match_percent, internet_content_percent, \
                        input_sentences, sources = \
                        analyze_text_similarity(input_text, search_results, similarity_threshold)
                        
                        # Display results in columns
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.metric("Final Plagiarism Score", f"{internet_content_percent:.1f}%")
                            st.metric("Total Input Words", str(get_text_stats(input_text)['word_count']))
                        
                        with col2:
                            content_label = "Content Originality"
                            if internet_content_percent > 80:
                                st.metric(content_label, f"{100 - internet_content_percent:.1f}%", delta="Low", delta_color="inverse")
                            elif internet_content_percent > 50:
                                st.metric(content_label, f"{100 - internet_content_percent:.1f}%", delta="Medium", delta_color="off")
                            else:
                                st.metric(content_label, f"{100 - internet_content_percent:.1f}%", delta="High")
                            st.metric("Matched Sources", f"{len(sources)}" if sources else "0")
                        
                        if results:
                            st.subheader("📚 Detailed Source Analysis")
                            
                            for source_link, matches in sorted(
                                sources.items(),
                                key=lambda x: max(m['adjusted_exact_score'] for m in x[1]),
                                reverse=True
                            ):
                                source_title = matches[0]['source_title']
                                max_score = max(m['adjusted_exact_score'] for m in matches)
                                
                                with st.expander(f"🔍 {source_title} - Match Score: {max_score:.1f}%"):
                                    st.markdown(f"**Source URL:** [{source_link}]({source_link})")
                                    st.markdown(f"**Number of Matches:** {len(matches)}")
                                    
                                    # Display text ratio information
                                    st.markdown("### Content Analysis")
                                    max_ratio = max(m['text_ratio'] for m in matches)
                                    st.markdown(f"**Text Length Ratio:** {max_ratio:.1f}%")
                                    
                                    st.markdown("### Matching Content")
                                    for match in sorted(matches, key=lambda x: x['adjusted_exact_score'], reverse=True):
                                        with st.container():
                                            score_color = (
                                                "🔴" if match['adjusted_exact_score'] >= 90 else
                                                "🔵"
                                            )
                                            
                                            st.markdown(f"""
                                            {score_color} **Adjusted Match Score: {match['adjusted_exact_score']:.1f}%**
                                            - Original Score: {match['exact_score']:.1f}%
                                            - Text Ratio: {match['text_ratio']:.1f}%
                                            - Input Words: {match['input_word_count']}
                                            - Source Words: {match['source_word_count']}
                                            
                                            **Your Text:**
                                            {match['input_text']}
                                            
                                            **Matched Text:**
                                            {match['matching_text']}
                                            
                                            **Match Details:**
                                            - Semantic Score: {match['semantic_score']:.1f}%
                                            - Type: {match['match_type'].title()}
                                            """)
                                            st.markdown("---")
                            
                            st.info(f"""
                            📊 Analysis Summary:
                            - Total Sources: {len(sources)}
                            - Total Matches: {len(results)}
                            - Overall Similarity: {internet_content_percent:.1f}%
                            - Average Text Ratio: {sum(r['text_ratio'] for r in results) / len(results):.1f}%
                            """)
                            
                            if st.button("📥 Download Detailed Report"):
                                report = generate_report(
                                    results,
                                    exact_match_percent,
                                    internet_content_percent,
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
    st.sidebar.write("""
    ### Advanced Analysis Features:
    1. Text Ratio Analysis:
       - Word count comparison
       - Character length analysis
       - Adjusted similarity scores

    2. Similarity Detection:
       - Exact Match Detection
       - Length-Aware Scoring

    3. Search Optimization:
       - Multi-query generation
       - Key term extraction
       - Intelligent source filtering

    ### Current Settings:
    - Similarity Threshold: {}%
    - Search Depth: {} sources
    """.format(similarity_threshold, search_depth))

if __name__ == "__main__":
    main()
