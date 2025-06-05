import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


def preprocess_text(text: str):
    sentences = sent_tokenize(text)
    stop_words = stopwords.words("english")
    processed_sentences = []
    for sentence in sentences:
        original_sentence = sentence  # Keep original sentence for summary

        # Remove special characters and convert to lowercase for processing
        sentence_for_tfidf = "".join(
            c for c in sentence if c.isalnum() or c.isspace())
        sentence_for_tfidf = sentence_for_tfidf.lower()

        words = word_tokenize(sentence_for_tfidf)
        words = [word for word in words if word not in stop_words]
        sentence_for_tfidf = " ".join(words)

        # Store a tuple of (processed_sentence, original_sentence)
        processed_sentences.append((sentence_for_tfidf, original_sentence))

    return processed_sentences


def generate_summary_improved(text: str, num_sentences: int = 3):
    # Get processed sentences and their original forms
    processed_and_original = preprocess_text(text)
    processed_sentences_for_tfidf = [item[0]
                                     for item in processed_and_original]
    original_sentences = [item[1] for item in processed_and_original]

    if not processed_sentences_for_tfidf:
        return ""  # Handle empty text

    # Filter out empty processed sentences before TF-IDF
    valid_indices = [i for i, s in enumerate(
        processed_sentences_for_tfidf) if s.strip()]
    if not valid_indices:
        return ""  # No valid sentences for TF-IDF

    filtered_processed_sentences = [
        processed_sentences_for_tfidf[i] for i in valid_indices]
    filtered_original_sentences = [
        original_sentences[i] for i in valid_indices]

    # Create TF-IDF vectorizer and matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_processed_sentences)

    # Calculate similarity scores between all sentences
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Sum the similarity scores for each sentence to get its importance
    # We sum up the similarities, excluding the similarity of a sentence to itself (which is 1)
    # Subtract 1 for self-similarity
    sentence_scores = np.sum(similarity_matrix, axis=1) - 1

    # Get the indices of the top 'num_sentences' most important sentences
    # using argsort and then slicing
    ranked_sentence_indices = np.argsort(sentence_scores)[
        ::-1]  # Descending order

    # Select the top sentences and maintain their original order
    summary_sentences = []
    selected_original_indices = []
    for idx in ranked_sentence_indices:
        if len(summary_sentences) < num_sentences:
            # Map back to original text index
            selected_original_indices.append(valid_indices[idx])

    # Sort selected sentences by their original appearance order for readability
    selected_original_indices.sort()

    for idx in selected_original_indices:
        summary_sentences.append(original_sentences[idx])

    return " ".join(summary_sentences)


# Example Usage with improved function
text = """Natural language processing (NLP) is a subfield of artificial intelligence, computer science, and linguistics. It focuses on the interactions between computers and human language. This allows computers to understand, interpret, and manipulate human language. NLP draws from many disciplines, including computer science and artificial intelligence. It also draws from linguistics, statistics, and machine learning. Historically, NLP was rule-based, relying on hand-crafted rules to parse and analyze text. With the rise of machine learning, especially deep learning, NLP has seen significant advancements. Modern NLP models are often data-driven, learning patterns from large text corpora. Applications of NLP include machine translation, sentiment analysis, spam detection, and chatbots. It's a rapidly evolving field with continuous research and development."""
summary = generate_summary_improved(text, num_sentences=3)
print("\n--- Improved Summary ---")
print(summary)
