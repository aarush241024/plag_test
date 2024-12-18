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

# Fair Metrics thresholds
EXACT_MATCH_THRESHOLD = 0.8
HIGH_SIMILARITY_THRESHOLD = 0.6
CITATION_THRESHOLD = 0.7

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
    """Enhanced search with multiple queries using SerpAPI's Google Search"""
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

def extract_statements(text):
    """
    Extract statements from text using sentence tokenization 
    and citation analysis
    """
    sentences = sent_tokenize(text)
    statements = []
    
    for sentence in sentences:
        clean_sentence = clean_text(sentence)
        if len(clean_sentence) < 10:
            continue
            
        has_citation = bool(re.search(r'\(\d{4}\)|\[\d+\]|et al\.', sentence))
        is_properly_cited = bool(re.search(r'\(([A-Za-z]+,?\s*)+\d{4}\)', sentence))
        
        statement = {
            'text': clean_sentence,
            'has_citation': has_citation,
            'is_properly_cited': is_properly_cited,
            'embedding': None
        }
        statements.append(statement)
    
    return statements

def calculate_statement_similarity(stmt1, stmt2):
    """Calculate semantic similarity between statements"""
    if stmt1['embedding'] is None:
        stmt1['embedding'] = models['semantic'].encode(stmt1['text'], convert_to_tensor=True)
    if stmt2['embedding'] is None:
        stmt2['embedding'] = models['semantic'].encode(stmt2['text'], convert_to_tensor=True)
    
    similarity = np.dot(stmt1['embedding'].cpu().numpy(), 
                       stmt2['embedding'].cpu().numpy())
    return similarity

def classify_statement_pair(test_stmt, control_stmt, similarity_threshold=0.8):
    """
    Classify statement relationships based on Fair Metrics:
    Q: Quoted (correctly cited)
    M: Misquoted (incorrectly cited)
    P: Plagiarized (uncited)
    N: Novel
    """
    similarity = calculate_statement_similarity(test_stmt, control_stmt)
    
    if similarity >= similarity_threshold:
        if test_stmt['is_properly_cited']:
            return 'Q', similarity
        elif test_stmt['has_citation']:
            return 'M', similarity
        else:
            return 'P', similarity
    else:
        return 'N', similarity

def calculate_fair_metrics(test_text, control_text):
    """Calculate Fair Metrics for plagiarism detection"""
    test_statements = extract_statements(test_text)
    control_statements = extract_statements(control_text)
    
    metrics = {
        'Q': 0,  # Quoted (correctly cited)
        'M': 0,  # Misquoted (incorrectly cited)
        'P': 0,  # Plagiarized (uncited)
        'N': 0,  # Novel
        'statement_details': []
    }
    
    for test_stmt in test_statements:
        best_match_type = 'N'
        best_match_score = 0
        best_match_stmt = None
        
        for control_stmt in control_statements:
            stmt_type, similarity = classify_statement_pair(test_stmt, control_stmt)
            
            if similarity > best_match_score:
                best_match_type = stmt_type
                best_match_score = similarity
                best_match_stmt = control_stmt
        
        metrics[best_match_type] += 1
        
        if best_match_stmt:
            metrics['statement_details'].append({
                'test_statement': test_stmt['text'],
                'control_statement': best_match_stmt['text'],
                'type': best_match_type,
                'similarity': best_match_score
            })
    
    metrics['S'] = metrics['M'] + metrics['Q'] + metrics['P']
    metrics['R'] = metrics['S'] + metrics['N']
    metrics['K'] = len(control_statements)
    
    try:
        F1 = metrics['Q'] / metrics['S'] if metrics['S'] > 0 else 0
        F2 = (metrics['Q'] - metrics['M']) / metrics['S'] if metrics['S'] > 0 else 0
        F3 = (metrics['Q'] - metrics['P']) / metrics['S'] if metrics['S'] > 0 else 0
        F4 = (metrics['Q'] - metrics['N']) / metrics['R'] if metrics['R'] > 0 else 0
        
        F1 = max(0, min(1, F1))
        F2 = max(-1, min(1, F2))
        F3 = max(-1, min(1, F3))
        F4 = max(-1, min(1, F4))
        
    except ZeroDivisionError:
        return 0, 0, 0, 0, metrics
    
    return F1, F2, F3, F4, metrics

def analyze_text_similarity(input_text, search_results, similarity_threshold):
    """Analyze text similarity using Fair Metrics"""
    cleaned_input = clean_text(input_text)
    results = []
    sources = defaultdict(list)
    
    total_sources = len(search_results)
    for idx, (title, snippet, link) in enumerate(search_results):
        try:
            progress = (idx + 1) / total_sources
            st.progress(progress, text=f"Analyzing source {idx + 1} of {total_sources}")
            
            source_text = clean_text(f"{title} {snippet}")
            if not source_text:
                continue
            
            F1, F2, F3, F4, metrics = calculate_fair_metrics(cleaned_input, source_text)
            
            plagiarism_score = (
                (F1 * 40) +
                (abs(F2) * -20) +
                (abs(F3) * -30) +
                (abs(F4) * -10)
            ) + 100
            
            if plagiarism_score >= similarity_threshold:
                match_type = (
                    'exact' if metrics['P'] / metrics['R'] > 0.7 else
                    'similar' if metrics['Q'] / metrics['R'] > 0.5 else
                    'improper' if metrics['M'] / metrics['R'] > 0.3 else
                    'low_similarity'
                )
                
                match_data = {
                    'input_text': cleaned_input,
                    'matching_text': source_text,
                    'plagiarism_score': plagiarism_score,
                    'metrics': metrics,
                    'F1': F1 * 100,
                    'F2': F2 * 100,
                    'F3': F3 * 100,
                    'F4': F4 * 100,
                    'match_type': match_type,
                    'source_title': title,
                    'source_link': link,
                    'statement_details': metrics['statement_details']
                }
                
                results.append(match_data)
                sources[link].append(match_data)
                
        except Exception as e:
            continue
    
    if results:
        total_statements = sum(len(r['metrics']['statement_details']) for r in results)
        total_plagiarized = sum(r['metrics']['P'] for r in results)
        total_misquoted = sum(r['metrics']['M'] for r in results)
        total_quoted = sum(r['metrics']['Q'] for r in results)
        
        exact_match_percent = (total_plagiarized / total_statements * 100) if total_statements > 0 else 0
        similar_content_percent = (total_quoted / total_statements * 100) if total_statements > 0 else 0
        improper_citation_percent = (total_misquoted / total_statements * 100) if total_statements > 0 else 0
        internet_content_percent = max(r['plagiarism_score'] for r in results)
    else:
        exact_match_percent = similar_content_percent = improper_citation_percent = internet_content_percent = 0
    
    return (results, exact_match_percent, similar_content_percent, 
            internet_content_percent, improper_citation_percent, [cleaned_input], sources)

def generate_report(results, exact_percent, similar_percent, plagiarism_percent, 
                   improper_percent, similarity_threshold, sources_analyzed, sources):
    """Generate detailed analysis report using Fair Metrics"""
    report = f"""Fair Metrics Plagiarism Analysis Report

Overall Metrics:
---------------
Plagiarism Score: {plagiarism_percent:.1f}%
Correctly Cited Content: {similar_percent:.1f}%
Uncited Similar Content: {exact_percent:.1f}%
Improper Citations: {improper_percent:.1f}%
Similarity Threshold: {similarity_threshold}%
Sources Analyzed: {sources_analyzed}

Analysis Method:
--------------
- Statement-level analysis
- Citation pattern recognition
- Semantic similarity computation
- Fair Metrics scoring (F1, F2, F3, F4)

Detailed Source Analysis:
----------------------"""

    for source_link, matches in sources.items():
        source_title = matches[0]['source_title']
        
        # Calculate source-level metrics
        total_statements = sum(len(m['statement_details']) for m in matches)
        total_quoted = sum(m['metrics']['Q'] for m in matches)
        total_misquoted = sum(m['metrics']['M'] for m in matches)
        total_plagiarized = sum(m['metrics']['P'] for m in matches)
        
        avg_F1 = sum(m['F1'] for m in matches) / len(matches)
        avg_F2 = sum(m['F2'] for m in matches) / len(matches)
        avg_F3 = sum(m['F3'] for m in matches) / len(matches)
        avg_F4 = sum(m['F4'] for m in matches) / len(matches)
        
        report += f"""

Source: {source_title}
URL: {source_link}
Fair Metrics:
- F1 (Citation Quality): {avg_F1:.1f}%
- F2 (Citation Accuracy): {avg_F2:.1f}%
- F3 (Uncited Content): {avg_F3:.1f}%
- F4 (Content Novelty): {avg_F4:.1f}%

Statement Analysis:
- Total Statements: {total_statements}
- Correctly Cited: {total_quoted}
- Incorrectly Cited: {total_misquoted}
- Uncited Similar: {total_plagiarized}

Matching Statements:"""

        for match in matches:
            for detail in match['statement_details']:
                report += f"""

Statement Type: {detail['type']}
Similarity: {detail['similarity']:.1f}
Input: {detail['test_statement']}
Match: {detail['control_statement']}
"""
    
    # Add recommendations section
    report += """

Recommendations:
--------------"""
    
    if plagiarism_percent > 80:
        report += """
- High levels of uncited content detected
- Significant revision needed
- Add proper citations for similar content
- Rephrase copied sections"""
    elif plagiarism_percent > 50:
        report += """
- Moderate levels of similar content found
- Review and add missing citations
- Consider rephrasing some sections
- Check citation formatting"""
    else:
        report += """
- Low similarity levels detected
- Add citations where needed
- Verify citation formatting
- Minor revisions recommended"""
    
    return report

def main():
    st.title("📝 Fair Metrics Plagiarism Detector")
    st.markdown("*Using Statement Analysis and Citation Recognition*")
    
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
        help="Adjust the minimum similarity threshold for detection"
    )
    
    search_depth = st.sidebar.slider(
        "Search Depth", 
        min_value=10, 
        max_value=100, 
        value=DEFAULT_SEARCH_DEPTH,
        help="Number of sources to analyze"
    )
    
    if st.button("🔍 Analyze Text"):
        if len(input_text) < 50:
            st.error("Please enter at least 50 characters for accurate analysis.")
        else:
            with st.spinner("🔄 Processing text..."):
                try:
                    st.info("📊 Analyzing text structure and citations...")
                    statements = extract_statements(input_text)
                    if statements:
                        st.success(f"Found {len(statements)} statements to analyze")
                    
                    st.info("🌐 Searching for similar content...")
                    search_results = get_search_results(input_text, search_depth)
                    
                    if search_results:
                        st.info(f"Found {len(search_results)} sources to analyze...")
                        
                        results, exact_match_percent, similar_content_percent, \
                        plagiarism_percent, improper_percent, input_sentences, sources = \
                        analyze_text_similarity(input_text, search_results, similarity_threshold)
                        
                        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                        
                        with col1:
                            st.metric("Fair Score (F1)", f"{sum(r['F1'] for r in results)/len(results):.1f}%" if results else "0%")
                            st.metric("Citation Quality", f"{similar_content_percent:.1f}%")
                        
                        with col2:
                            st.metric("Citation Issues (F2)", f"{sum(r['F2'] for r in results)/len(results):.1f}%" if results else "0%")
                            st.metric("Improper Citations", f"{improper_percent:.1f}%")
                        
                        with col3:
                            plagiarism_label = "Plagiarism Score 🌐"
                            if plagiarism_percent > 80:
                                st.metric(plagiarism_label, f"{plagiarism_percent:.1f}%", delta="High", delta_color="inverse")
                            elif plagiarism_percent > 50:
                                st.metric(plagiarism_label, f"{plagiarism_percent:.1f}%", delta="Medium", delta_color="off")
                            else:
                                st.metric(plagiarism_label, f"{plagiarism_percent:.1f}%", delta="Low")
                            st.metric("Uncited Content", f"{exact_match_percent:.1f}%")
                        
                        with col4:
                            st.metric("Content Novelty (F4)", f"{sum(r['F4'] for r in results)/len(results):.1f}%" if results else "0%")
                            st.metric("Sources Analyzed", str(len(sources)))
                        
                        if results:
                            st.subheader("📚 Detailed Source Analysis")
                            
                            for source_link, matches in sorted(
                                sources.items(),
                                key=lambda x: max(m['plagiarism_score'] for m in x[1]),
                                reverse=True
                            ):
                                source_title = matches[0]['source_title']
                                max_score = max(m['plagiarism_score'] for m in matches)
                                
                                with st.expander(f"🔍 {source_title} - Plagiarism Score: {max_score:.1f}%"):
                                    st.markdown(f"**Source URL:** [{source_link}]({source_link})")
                                    
                                    # Display Fair Metrics for this source
                                    avg_metrics = {
                                        'F1': sum(m['F1'] for m in matches) / len(matches),
                                        'F2': sum(m['F2'] for m in matches) / len(matches),
                                        'F3': sum(m['F3'] for m in matches) / len(matches),
                                        'F4': sum(m['F4'] for m in matches) / len(matches)
                                    }
                                    
                                    st.markdown(f"""
                                    ### Fair Metrics Analysis
                                    - F1 (Citation Quality): {avg_metrics['F1']:.1f}%
                                    - F2 (Citation Accuracy): {avg_metrics['F2']:.1f}%
                                    - F3 (Uncited Content): {avg_metrics['F3']:.1f}%
                                    - F4 (Content Novelty): {avg_metrics['F4']:.1f}%
                                    """)
                                    
                                    st.markdown("### Statement Analysis")
                                    for match in matches:
                                        for detail in match['statement_details']:
                                            with st.container():
                                                match_color = (
                                                    "🔴" if detail['type'] == 'P' else
                                                    "🟡" if detail['type'] == 'M' else
                                                    "🟢" if detail['type'] == 'Q' else
                                                    "⚪"
                                                )
                                                
                                                match_type = {
                                                    'P': 'Plagiarized',
                                                    'M': 'Misquoted',
                                                    'Q': 'Properly Cited',
                                                    'N': 'Novel'
                                                }[detail['type']]
                                                
                                                st.markdown(f"""
                                                {match_color} **Match Type: {match_type}**
                                                
                                                **Your Text:**
                                                {detail['test_statement']}
                                                
                                                **Matched Text:**
                                                {detail['control_statement']}
                                                
                                                **Similarity:** {detail['similarity']:.1f}
                                                """)
                                                st.markdown("---")
                            
                            match_level = "High" if plagiarism_percent > 80 else "Moderate" if plagiarism_percent > 50 else "Low"
                            
                            if st.button("📥 Download Detailed Report"):
                                report = generate_report(
                                    results,
                                    exact_match_percent,
                                    similar_content_percent,
                                    plagiarism_percent,
                                    improper_percent,
                                    similarity_threshold,
                                    len(search_results),
                                    sources
                                )
                                
                                st.download_button(
                                    "📄 Save Report",
                                    report,
                                    "fair_metrics_report.txt",
                                    "text/plain"
                                )
                        else:
                            st.success(f"✅ No significant matches found above {similarity_threshold}% threshold!")
                    else:
                        st.warning("No search results found. Try modifying your text or check your API key.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("If you're seeing API errors, please check your SerpAPI key and usage limits.")

    st.sidebar.title("ℹ️ About")
    st.sidebar.write("""
    ### Fair Metrics Analysis:
    1. Statement Analysis:
       - Citation detection
       - Statement classification
       - Semantic similarity

    2. Fair Metrics:
       - F1: Citation Quality [0,1]
       - F2: Citation Accuracy [-1,+1]
       - F3: Uncited Content [-1,+1]
       - F4: Content Novelty [-1,+1]

    3. Classification:
       - Q: Properly cited content
       - M: Incorrectly cited content
       - P: Uncited similar content
       - N: Novel content

    ### Current Settings:
    - Similarity Threshold: {}%
    - Search Depth: {} sources
    """.format(similarity_threshold, search_depth))

if __name__ == "__main__":
    main()