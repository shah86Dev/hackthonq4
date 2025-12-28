import logging
from typing import List, Dict, Tuple
from src.services.generation_service import GenerationService
from src.services.retrieval_service import RetrievalService
from src.agents.coordinator_agent import CoordinatorAgent
from sqlalchemy.orm import Session
from src.models.query_log import QueryLog
from src.evaluation.metrics import calculate_response_quality_metrics
import time
import json

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Automated evaluation framework for RAG chatbot responses
    """

    def __init__(self, db: Session):
        self.db = db
        self.generation_service = GenerationService()
        self.retrieval_service = RetrievalService()
        self.coordinator_agent = CoordinatorAgent()
        self.evaluation_results = []

    def evaluate_response(self, question: str, expected_answer: str, book_id: str, selected_text: str = None) -> Dict:
        """
        Evaluate a single response against the expected answer
        """
        try:
            start_time = time.time()

            # Get the actual response from the system
            response_data = self.coordinator_agent.process_query(
                db=self.db,
                book_id=book_id,
                question=question,
                selected_text=selected_text
            )

            response_time = time.time() - start_time

            # Calculate metrics comparing actual response to expected answer
            metrics = calculate_response_quality_metrics(
                actual_response=response_data["answer"],
                expected_response=expected_answer,
                context_chunks=response_data.get("context_chunks_used", [])
            )

            # Combine response data with metrics
            evaluation_result = {
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": response_data["answer"],
                "book_id": book_id,
                "selected_text": selected_text,
                "response_time": response_time,
                "metrics": metrics,
                "context_chunks_used": response_data.get("context_chunks_used", []),
                "confidence_score": response_data.get("confidence_score", 0.0),
                "mode": response_data.get("mode", "full-book")
            }

            self.evaluation_results.append(evaluation_result)
            logger.info(f"Evaluation completed for question: {question[:50]}...")

            return evaluation_result
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return {
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": f"Error: {str(e)}",
                "book_id": book_id,
                "selected_text": selected_text,
                "response_time": 0,
                "metrics": {"error": str(e)},
                "context_chunks_used": [],
                "confidence_score": 0.0,
                "mode": "error"
            }

    def evaluate_batch(self, test_cases: List[Dict]) -> List[Dict]:
        """
        Evaluate multiple test cases in batch
        """
        results = []
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")
            result = self.evaluate_response(
                question=test_case["question"],
                expected_answer=test_case["expected_answer"],
                book_id=test_case["book_id"],
                selected_text=test_case.get("selected_text")
            )
            results.append(result)
        return results

    def compare_ground_truth(self, test_cases: List[Dict]) -> Dict:
        """
        Compare system responses against ground truth answers
        """
        results = self.evaluate_batch(test_cases)

        # Calculate overall metrics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r["metrics"].get("similarity_score", 0) >= 0.7)
        avg_response_time = sum(r["response_time"] for r in results) / total_tests if total_tests > 0 else 0
        avg_similarity = sum(r["metrics"].get("similarity_score", 0) for r in results) / total_tests if total_tests > 0 else 0
        avg_factual_accuracy = sum(r["metrics"].get("factual_accuracy", 0) for r in results) / total_tests if total_tests > 0 else 0

        summary = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "average_response_time": avg_response_time,
            "average_similarity": avg_similarity,
            "average_factual_accuracy": avg_factual_accuracy,
            "results": results
        }

        logger.info(f"Evaluation summary: {summary['success_rate']:.2%} success rate, "
                   f"{summary['average_response_time']:.2f}s avg response time, "
                   f"{summary['average_similarity']:.2f} avg similarity")

        return summary

    def run_comprehensive_evaluation(self, test_dataset: List[Dict]) -> Dict:
        """
        Run comprehensive evaluation with multiple metrics
        """
        logger.info(f"Starting comprehensive evaluation with {len(test_dataset)} test cases")

        # Run the comparison against ground truth
        comparison_results = self.compare_ground_truth(test_dataset)

        # Additional analysis
        detailed_analysis = self._perform_detailed_analysis(comparison_results["results"])

        # Compile final report
        final_report = {
            "summary": comparison_results,
            "detailed_analysis": detailed_analysis,
            "timestamp": time.time(),
            "total_evaluations": len(test_dataset)
        }

        logger.info("Comprehensive evaluation completed")
        return final_report

    def _perform_detailed_analysis(self, results: List[Dict]) -> Dict:
        """
        Perform detailed analysis of evaluation results
        """
        analysis = {
            "response_times": [r["response_time"] for r in results],
            "similarity_scores": [r["metrics"].get("similarity_score", 0) for r in results],
            "factual_accuracies": [r["metrics"].get("factual_accuracy", 0) for r in results],
            "confidence_scores": [r["confidence_score"] for r in results],
            "high_similarity_count": sum(1 for r in results if r["metrics"].get("similarity_score", 0) >= 0.8),
            "low_similarity_count": sum(1 for r in results if r["metrics"].get("similarity_score", 0) < 0.5),
            "high_factual_accuracy_count": sum(1 for r in results if r["metrics"].get("factual_accuracy", 0) >= 0.8),
            "low_factual_accuracy_count": sum(1 for r in results if r["metrics"].get("factual_accuracy", 0) < 0.5),
        }

        # Calculate percentiles
        analysis["response_time_percentiles"] = self._calculate_percentiles(analysis["response_times"], [50, 90, 95])
        analysis["similarity_percentiles"] = self._calculate_percentiles(analysis["similarity_scores"], [50, 90, 95])

        return analysis

    def _calculate_percentiles(self, values: List[float], percentiles: List[int]) -> Dict[int, float]:
        """
        Calculate percentiles for a list of values
        """
        if not values:
            return {p: 0.0 for p in percentiles}

        sorted_values = sorted(values)
        n = len(sorted_values)

        percentile_values = {}
        for p in percentiles:
            index = int((p / 100) * n)
            index = min(index, n - 1)  # Ensure index is within bounds
            percentile_values[p] = sorted_values[index]

        return percentile_values

    def save_evaluation_report(self, report: Dict, filename: str = "evaluation_report.json"):
        """
        Save evaluation report to a file
        """
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Evaluation report saved to {filename}")

    def get_evaluation_accuracy(self) -> float:
        """
        Get the overall accuracy of the evaluation system
        """
        if not self.evaluation_results:
            return 0.0

        successful_evaluations = sum(
            1 for result in self.evaluation_results
            if result["metrics"].get("factual_accuracy", 0) >= 0.7
        )
        return successful_evaluations / len(self.evaluation_results)