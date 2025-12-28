import logging
from typing import List, Dict, Any
from difflib import SequenceMatcher
import re

logger = logging.getLogger(__name__)

def calculate_response_quality_metrics(actual_response: str, expected_response: str, context_chunks: List[Dict] = None) -> Dict[str, Any]:
    """
    Calculate various quality metrics for the response compared to the expected answer
    """
    try:
        # Calculate similarity between actual and expected responses
        similarity_score = calculate_similarity(actual_response, expected_response)

        # Calculate factual accuracy based on how well the response matches expected content
        factual_accuracy = calculate_factual_accuracy(actual_response, expected_response)

        # Calculate citation quality based on context chunks used
        citation_quality = calculate_citation_quality(context_chunks)

        # Calculate response completeness
        response_completeness = calculate_response_completeness(actual_response, expected_response)

        # Calculate hallucination detection (responses that contain information not in context)
        hallucination_score = calculate_hallucination_score(actual_response, context_chunks)

        # Calculate overall quality score
        overall_quality = calculate_overall_quality(similarity_score, factual_accuracy, citation_quality, hallucination_score)

        metrics = {
            "similarity_score": similarity_score,
            "factual_accuracy": factual_accuracy,
            "citation_quality": citation_quality,
            "response_completeness": response_completeness,
            "hallucination_score": hallucination_score,
            "overall_quality": overall_quality,
            "metric_calculation_timestamp": __import__('time').time()
        }

        return metrics
    except Exception as e:
        logger.error(f"Error calculating response quality metrics: {e}")
        return {
            "similarity_score": 0.0,
            "factual_accuracy": 0.0,
            "citation_quality": 0.0,
            "response_completeness": 0.0,
            "hallucination_score": 1.0,  # Default to high hallucination on error
            "overall_quality": 0.0,
            "error": str(e)
        }

def calculate_similarity(response1: str, response2: str) -> float:
    """
    Calculate similarity between two responses using sequence matching
    """
    if not response1 or not response2:
        return 0.0

    # Normalize the responses for comparison
    normalized_resp1 = normalize_text(response1)
    normalized_resp2 = normalize_text(response2)

    # Calculate similarity ratio
    similarity = SequenceMatcher(None, normalized_resp1, normalized_resp2).ratio()
    return max(0.0, min(1.0, similarity))  # Ensure value is between 0 and 1

def calculate_factual_accuracy(actual_response: str, expected_response: str) -> float:
    """
    Calculate factual accuracy by checking how much of the expected information is present in the actual response
    """
    if not actual_response or not expected_response:
        return 0.0

    # Normalize both responses
    normalized_actual = normalize_text(actual_response)
    normalized_expected = normalize_text(expected_response)

    # Extract key phrases from expected response
    expected_phrases = extract_key_phrases(normalized_expected)

    # Count how many expected phrases are present in the actual response
    matched_phrases = 0
    for phrase in expected_phrases:
        if phrase.lower() in normalized_actual.lower():
            matched_phrases += 1

    # Calculate accuracy as ratio of matched phrases
    accuracy = matched_phrases / len(expected_phrases) if expected_phrases else 0.0
    return max(0.0, min(1.0, accuracy))

def calculate_citation_quality(context_chunks: List[Dict]) -> float:
    """
    Calculate citation quality based on the quality of context chunks used
    """
    if not context_chunks:
        return 0.0

    # For now, we'll calculate based on the number of chunks and their relevance scores
    total_chunks = len(context_chunks)
    if total_chunks == 0:
        return 0.0

    # Calculate average relevance score of chunks
    total_relevance = sum(chunk.get("score", 0.0) for chunk in context_chunks)
    avg_relevance = total_relevance / total_chunks

    # Calculate citation density (how many relevant chunks were used)
    relevant_chunks = sum(1 for chunk in context_chunks if chunk.get("score", 0.0) > 0.5)
    citation_density = relevant_chunks / total_chunks

    # Combine metrics for citation quality
    citation_quality = (avg_relevance * 0.6) + (citation_density * 0.4)
    return max(0.0, min(1.0, citation_quality))

def calculate_response_completeness(actual_response: str, expected_response: str) -> float:
    """
    Calculate how complete the response is compared to the expected response
    """
    if not actual_response or not expected_response:
        return 0.0

    # Calculate word overlap
    actual_words = set(normalize_text(actual_response).split())
    expected_words = set(normalize_text(expected_response).split())

    if not expected_words:
        return 0.0

    # Calculate overlap
    overlap = len(actual_words.intersection(expected_words))
    completeness = overlap / len(expected_words)

    # Also consider length similarity
    length_similarity = min(len(actual_response), len(expected_response)) / max(len(actual_response), len(expected_response))

    # Combine both metrics
    completeness_score = (completeness * 0.7) + (length_similarity * 0.3)
    return max(0.0, min(1.0, completeness_score))

def calculate_hallucination_score(actual_response: str, context_chunks: List[Dict]) -> float:
    """
    Calculate hallucination score (lower is better)
    """
    if not context_chunks:
        # If no context chunks, assume higher chance of hallucination
        return 0.8

    # Extract all text from context chunks
    context_text = " ".join([chunk.get("text", "") for chunk in context_chunks])
    context_text = normalize_text(context_text)

    # Check for claims in the actual response that are not supported by context
    actual_sentences = [s.strip() for s in actual_response.split('.') if s.strip()]
    unsupported_sentences = 0

    for sentence in actual_sentences:
        normalized_sentence = normalize_text(sentence)
        if not is_sentence_supported(normalized_sentence, context_text):
            unsupported_sentences += 1

    # Calculate hallucination score (0 = no hallucination, 1 = all hallucination)
    hallucination_score = unsupported_sentences / len(actual_sentences) if actual_sentences else 0.0
    return max(0.0, min(1.0, hallucination_score))

def calculate_overall_quality(similarity_score: float, factual_accuracy: float, citation_quality: float, hallucination_score: float) -> float:
    """
    Calculate overall quality score combining all metrics
    """
    # Weight the metrics appropriately
    # Similarity and factual accuracy are most important
    # Citation quality is important for RAG systems
    # Lower hallucination score is better, so we subtract it
    overall_score = (
        (similarity_score * 0.3) +
        (factual_accuracy * 0.3) +
        (citation_quality * 0.2) +
        ((1 - hallucination_score) * 0.2)  # Invert hallucination score
    )
    return max(0.0, min(1.0, overall_score))

def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and special characters
    """
    if not text:
        return ""

    # Convert to lowercase and remove extra whitespace
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    # Remove special characters but keep spaces and alphanumeric
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    return normalized.strip()

def extract_key_phrases(text: str) -> List[str]:
    """
    Extract key phrases from text (simplified approach)
    """
    if not text:
        return []

    # Split by sentences and return non-empty phrases
    phrases = [phrase.strip() for phrase in text.split('.') if phrase.strip()]
    return phrases

def is_sentence_supported(sentence: str, context: str) -> bool:
    """
    Check if a sentence is supported by the context
    """
    if not sentence or not context:
        return False

    # Simple check: see if key terms from the sentence appear in the context
    sentence_words = set(sentence.split())
    context_words = set(context.split())

    # If at least 50% of the words in the sentence appear in the context, consider it supported
    if not sentence_words:
        return True  # Empty sentence is considered supported

    common_words = sentence_words.intersection(context_words)
    support_ratio = len(common_words) / len(sentence_words)

    # For now, we'll say 30% overlap is sufficient to consider the sentence supported
    return support_ratio >= 0.3

def calculate_response_quality_for_query_log(query_log: Dict) -> Dict[str, Any]:
    """
    Calculate quality metrics for a query log entry
    """
    # This would typically compare the query log's answer against some expected answer
    # For now, we'll return basic metrics based on the available data
    return {
        "latency_score": min(1.0, 2.0 / max(0.001, query_log.get("latency", 1.0))),  # Higher score for lower latency
        "confidence_score": query_log.get("confidence_score", 0.0),
        "tokens_efficiency": query_log.get("tokens_used", 0) / max(1, len(query_log.get("answer", "")) or 1),
        "quality_assessment_timestamp": __import__('time').time()
    }