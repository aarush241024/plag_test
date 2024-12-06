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

DEFAULT_SIMILARITY_THRESHOLD = 60
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
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    return ' '.join(words)

def extract_key_terms(text, n_terms=8):
    """Extract key terms from text using TF-IDF"""
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Fit TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_text])
    except ValueError:
        return []

    # Get feature names and scores
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    
    # Sort terms by score
    term_scores = [(term, score) for term, score in zip(feature_names, scores)]
    term_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N terms
    return [term for term, _ in term_scores[:n_terms]]

def generate_search_queries(text):
    """Generate multiple search queries from input text"""
    sentences = sent_tokenize(text)
    queries = []
    
    # Extract key terms from the entire text
    key_terms = extract_key_terms(text)
    if key_terms:
        # Add a query with just key terms
        queries.append(' '.join(key_terms[:MAX_QUERY_TERMS]))
    
    # Generate queries from sentences
    for sentence in sentences:
        terms = extract_key_terms(sentence, n_terms=MAX_QUERY_TERMS)
        if len(terms) >= MIN_QUERY_TERMS:
            queries.append(' '.join(terms))
    
    # Deduplicate and limit queries
    return list(set(queries))[:5]  # Limit to top 5 unique queries

def get_search_results(text, num_results=100):
    """Enhanced search with multiple queries"""
    all_results = []
    seen_links = set()
    
    # Generate multiple search queries
    queries = generate_search_queries(text)
    
    for query in queries:
        try:
            search = GoogleSearch({
                "q": query,
                "api_key": os.getenv("SERPAPI_KEY"),
                "engine": "google",
                "num": num_results // len(queries),  # Distribute results across queries
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
            
    return all_results[:num_results]  # Limit to requested number of results

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
    """Calculate similarity score between two texts with improved sequence matching"""
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    if text1 == text2:
        return 100.0

    # Break into words and word pairs (bigrams)
    def get_ngrams(text, n):
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    words1 = text1.split()
    words2 = text2.split()
    
    # Get n-grams for different sizes
    unigrams1 = set(words1)
    unigrams2 = set(words2)
    bigrams1 = set(get_ngrams(text1, 2))
    bigrams2 = set(get_ngrams(text2, 2))
    trigrams1 = set(get_ngrams(text1, 3))
    trigrams2 = set(get_ngrams(text2, 3))
    
    # Find longest common subsequences
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
    
    # Calculate scores for different levels of matching
    unigram_overlap = len(unigrams1 & unigrams2) / len(unigrams1 | unigrams2) * 100
    bigram_overlap = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2) * 100 if bigrams1 and bigrams2 else 0
    trigram_overlap = len(trigrams1 & trigrams2) / len(trigrams1 | trigrams2) * 100 if trigrams1 and trigrams2 else 0
    
    # Calculate sequence coverage
    sequence_words = set(' '.join(common_sequences).split())
    sequence_coverage = len(sequence_words) / max(len(unigrams1), len(unigrams2)) * 100
    
    # Calculate containment for longer phrases
    total_words = len(words1) + len(words2)
    sequence_length = sum(len(seq.split()) for seq in common_sequences)
    sequence_ratio = (sequence_length / total_words) * 200  # Multiply by 2 to normalize to 100
    
    # Weighted final score with emphasis on longer matching sequences
    exact_match_score = (
        unigram_overlap * 0.2 +      # Individual word matches
        bigram_overlap * 0.2 +       # Two-word phrase matches
        trigram_overlap * 0.2 +      # Three-word phrase matches
        sequence_coverage * 0.2 +     # Coverage of longest matches
        sequence_ratio * 0.2         # Proportion of text in matches
    )
    
    return min(exact_match_score, 100.0)

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
    results = []
    sources = defaultdict(list)
    
    # Vectorize input text once
    input_vector = models['semantic'].encode(cleaned_input, convert_to_tensor=True)
    
    total_sources = len(search_results)
    for idx, (title, snippet, link) in enumerate(search_results):
        try:
            progress = (idx + 1) / total_sources
            st.progress(progress, text=f"Analyzing source {idx + 1} of {total_sources}")
            
            source_text = clean_text(f"{title} {snippet}")
            if not source_text:
                continue
            
            # Calculate similarities
            exact_score = calculate_exact_match_score(cleaned_input, source_text)
            semantic_score = calculate_semantic_similarity(cleaned_input, source_text)
            
            # Only process if similarity is above threshold
            if exact_score >= similarity_threshold or semantic_score >= similarity_threshold:
                match_type = (
                    'exact' if exact_score >= 90 else
                    'similar' if exact_score >= 70 else
                    'paraphrase' if semantic_score >= 80 and exact_score < 70 else
                    'low_similarity'
                )
                
                match_data = {
                    'input_text': cleaned_input,
                    'matching_text': source_text,
                    'exact_score': exact_score,
                    'semantic_score': semantic_score,
                    'match_type': match_type,
                    'source_title': title,
                    'source_link': link
                }
                results.append(match_data)
                sources[link].append(match_data)
                        
        except Exception as e:
            continue

    # Calculate overall percentages based on highest matches
    if results:
        max_exact = max(r['exact_score'] for r in results)
        max_semantic = max(r['semantic_score'] for r in results)
        
        exact_matches = sum(1 for r in results if r['exact_score'] >= 90)
        similar_matches = sum(1 for r in results if 70 <= r['exact_score'] < 90)
        paraphrase_matches = sum(1 for r in results 
                               if r['semantic_score'] >= 80 
                               and r['exact_score'] < 70)
        
        total_matches = len(results)
        exact_match_percent = max_exact
        similar_content_percent = (similar_matches / total_matches * 100) if total_matches > 0 else 0
        paraphrase_percent = (paraphrase_matches / total_matches * 100) if total_matches > 0 else 0
        internet_content_percent = max(max_exact, max_semantic)
    else:
        exact_match_percent = similar_content_percent = paraphrase_percent = internet_content_percent = 0
    
    return results, exact_match_percent, similar_content_percent, internet_content_percent, paraphrase_percent, [cleaned_input], sources

def generate_report(results, exact_percent, similar_percent, internet_percent, 
                   paraphrase_percent, similarity_threshold, sources_analyzed, sources):
    """Generate a detailed analysis report"""
    report = f"""Plagiarism Analysis Report

Overall Metrics:
---------------
Exact Matches: {exact_percent:.1f}%
Similar Content: {similar_percent:.1f}%
Internet Content: {internet_percent:.1f}%
Potential Paraphrasing: {paraphrase_percent:.1f}%
Similarity Threshold: {similarity_threshold}%
Sources Analyzed: {sources_analyzed}

Detailed Analysis:
----------------"""

    # Add vectorization information
    report += """
Vectorization Analysis:
- Full text vectorization using SentenceTransformer
- TF-IDF analysis for key term extraction
- N-gram analysis (1-3 grams)
- Semantic similarity computation
"""

    report += "\nSource Analysis:\n--------------"

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
  Match Type: {match['match_type']}
"""
    
    # Add recommendations section
    report += """
Recommendations:
--------------
"""
    if internet_percent > INTERNET_CONTENT_HIGH:
        report += "- High internet content detected. Significant revision recommended.\n"
        report += "- Consider rewriting sections with high similarity scores.\n"
    elif internet_percent > INTERNET_CONTENT_MODERATE:
        report += "- Moderate internet content detected. Some revision may be needed.\n"
        report += "- Review and rephrase sections with exact matches.\n"
    else:
        report += "- Low internet content detected. Minor revisions may improve originality.\n"
    
    return report

def main():
    st.title("📝 Enhanced Plagiarism Detector")
    st.markdown("*With Advanced Vectorization and Internet Content Detection*")
    
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
                    # First phase: Vectorization and key term extraction
                    st.info("📊 Vectorizing input text and extracting key terms...")
                    key_terms = extract_key_terms(input_text)
                    if key_terms:
                        st.success(f"Key terms identified: {', '.join(key_terms)}")
                    
                    # Second phase: Search
                    st.info("🌐 Searching for similar content...")
                    search_results = get_search_results(input_text, search_depth)
                    
                    if search_results:
                        st.info(f"Found {len(search_results)} sources to analyze...")
                        
                        # Third phase: Analysis
                        results, exact_match_percent, similar_content_percent, \
                        internet_content_percent, paraphrase_percent, input_sentences, sources = \
                        analyze_text_similarity(input_text, search_results, similarity_threshold)
                        
                        # Display results in columns
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
                            st.metric("Matched Sources", f"{len(sources)}" if sources else "0")
                        
                        with col4:
                            st.metric("Possible Paraphrasing", f"{paraphrase_percent:.1f}%")
                            st.metric("Sources Analyzed", str(len(sources)))
                        
                        if results:
                            st.subheader("📚 Detailed Source Analysis")
                            
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
                                            {match['input_text']}
                                            
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
    st.sidebar.write("""
    ### Advanced Analysis Features:
    1. Text Vectorization:
       - Full document vectorization
       - TF-IDF key term extraction
       - N-gram analysis (1-3 grams)

    2. Similarity Detection:
       - Exact Match: ≥90% similarity
       - High Similarity: 70-90%
       - Semantic Match: Based on meaning
       - Paraphrase Detection

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
