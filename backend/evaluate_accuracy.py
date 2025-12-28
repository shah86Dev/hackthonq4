#!/usr/bin/env python3
"""
Comprehensive evaluation script to validate response accuracy >95% for Book-Embedded RAG Chatbot
"""

import asyncio
import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import calculate_response_quality_metrics
from tests.data.test_dataset import TEST_DATASET, EXTENDED_TEST_DATASET
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database import Base
import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Data class for evaluation results"""
    test_id: str
    question: str
    expected_answer: str
    actual_answer: str
    similarity_score: float
    factual_accuracy: float
    response_time: float
    is_correct: bool
    confidence_score: float


class AccuracyEvaluator:
    """
    Class to evaluate the accuracy of the RAG chatbot responses
    """

    def __init__(self, db_url: str = "sqlite:///./test.db"):
        # Set up database
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Initialize evaluator
        self.evaluator = Evaluator(self.SessionLocal())

    def run_comprehensive_evaluation(self) -> Dict:
        """
        Run comprehensive evaluation with multiple test datasets
        """
        logger.info("Starting comprehensive accuracy evaluation...")

        # Combine all test datasets
        all_tests = TEST_DATASET + EXTENDED_TEST_DATASET

        results = []
        correct_count = 0
        total_time = 0

        logger.info(f"Running evaluation on {len(all_tests)} test cases...")

        for i, test_case in enumerate(all_tests):
            logger.info(f"Processing test case {i+1}/{len(all_tests)}: {test_case['question'][:50]}...")

            try:
                # Run evaluation for this test case
                result = self.evaluator.evaluate_response(
                    question=test_case["question"],
                    expected_answer=test_case["expected_answer"],
                    book_id=test_case["book_id"],
                    selected_text=test_case.get("selected_text")
                )

                # Calculate if this response is correct (based on similarity and factual accuracy)
                similarity_score = result["metrics"].get("similarity_score", 0)
                factual_accuracy = result["metrics"].get("factual_accuracy", 0)

                # Consider response correct if both similarity and factual accuracy are above 70%
                is_correct = similarity_score >= 0.7 and factual_accuracy >= 0.7

                if is_correct:
                    correct_count += 1

                total_time += result["response_time"]

                eval_result = EvaluationResult(
                    test_id=f"test_{i+1}",
                    question=test_case["question"],
                    expected_answer=test_case["expected_answer"],
                    actual_answer=result["actual_answer"],
                    similarity_score=similarity_score,
                    factual_accuracy=factual_accuracy,
                    response_time=result["response_time"],
                    is_correct=is_correct,
                    confidence_score=result["confidence_score"]
                )

                results.append(eval_result)

                logger.info(f"Test {i+1}: Correct={is_correct}, Similarity={similarity_score:.2f}, Factual Acc={factual_accuracy:.2f}, Time={result['response_time']:.2f}s")

            except Exception as e:
                logger.error(f"Error evaluating test case {i+1}: {e}")
                # Mark as incorrect if there was an error
                results.append(EvaluationResult(
                    test_id=f"test_{i+1}",
                    question=test_case["question"],
                    expected_answer=test_case["expected_answer"],
                    actual_answer=f"ERROR: {str(e)}",
                    similarity_score=0.0,
                    factual_accuracy=0.0,
                    response_time=0.0,
                    is_correct=False,
                    confidence_score=0.0
                ))

        # Calculate overall accuracy
        total_tests = len(results)
        accuracy_percentage = (correct_count / total_tests) * 100 if total_tests > 0 else 0
        avg_response_time = total_time / total_tests if total_tests > 0 else 0

        # Prepare evaluation summary
        summary = {
            "total_tests": total_tests,
            "correct_responses": correct_count,
            "accuracy_percentage": accuracy_percentage,
            "average_response_time": avg_response_time,
            "detailed_results": [
                {
                    "test_id": r.test_id,
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "actual_answer": r.actual_answer,
                    "similarity_score": r.similarity_score,
                    "factual_accuracy": r.factual_accuracy,
                    "response_time": r.response_time,
                    "is_correct": r.is_correct,
                    "confidence_score": r.confidence_score
                } for r in results
            ],
            "timestamp": time.time(),
            "requirements_met": accuracy_percentage >= 95
        }

        logger.info(f"Evaluation completed. Accuracy: {accuracy_percentage:.2f}% ({correct_count}/{total_tests} correct)")
        logger.info(f"Average response time: {avg_response_time:.2f}s")
        logger.info(f"Requirements met (95% accuracy): {summary['requirements_met']}")

        return summary

    def generate_evaluation_report(self, summary: Dict, output_file: str = "evaluation_report.json"):
        """
        Generate a detailed evaluation report
        """
        report = {
            "evaluation_summary": {
                "total_tests": summary["total_tests"],
                "correct_responses": summary["correct_responses"],
                "accuracy_percentage": summary["accuracy_percentage"],
                "average_response_time": summary["average_response_time"],
                "requirements_met": summary["requirements_met"],
                "pass_threshold": 95.0,
                "timestamp": summary["timestamp"]
            },
            "detailed_results": summary["detailed_results"],
            "recommendations": self._generate_recommendations(summary)
        }

        # Save report to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {output_file}")
        return report

    def _generate_recommendations(self, summary: Dict) -> List[str]:
        """
        Generate recommendations based on evaluation results
        """
        recommendations = []

        if summary["accuracy_percentage"] < 95:
            recommendations.append(
                "Accuracy is below the 95% threshold. Consider improving the RAG pipeline, "
                "fine-tuning the model, or enhancing the retrieval mechanism."
            )
        else:
            recommendations.append(
                "Accuracy meets the 95% threshold requirement. The system performs well."
            )

        if summary["average_response_time"] > 2.0:  # More than 2 seconds
            recommendations.append(
                "Average response time is high (>2s). Consider optimizing the retrieval process, "
                "implementing caching, or using faster models for initial screening."
            )
        else:
            recommendations.append(
                "Response time is acceptable (<2s average)."
            )

        # Analyze incorrect responses for patterns
        incorrect_responses = [r for r in summary["detailed_results"] if not r["is_correct"]]
        if len(incorrect_responses) > 0:
            incorrect_ratio = len(incorrect_responses) / len(summary["detailed_results"])
            if incorrect_ratio > 0.1:  # More than 10% incorrect
                recommendations.append(
                    f"High error rate ({incorrect_ratio:.1%}). Investigate common failure patterns "
                    "in the incorrectly answered questions."
                )

        return recommendations


def main():
    """
    Main function to run the evaluation
    """
    logger.info("Starting Book-Embedded RAG Chatbot accuracy evaluation...")

    # Get database URL from environment or use default
    db_url = os.getenv("DATABASE_URL", "sqlite:///./test.db")

    # Initialize evaluator
    evaluator = AccuracyEvaluator(db_url)

    try:
        # Run comprehensive evaluation
        summary = evaluator.run_comprehensive_evaluation()

        # Generate detailed report
        report = evaluator.generate_evaluation_report(summary)

        # Print summary to console
        print("\n" + "="*60)
        print("ACCURACY EVALUATION RESULTS")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Correct Responses: {summary['correct_responses']}")
        print(f"Accuracy: {summary['accuracy_percentage']:.2f}%")
        print(f"Average Response Time: {summary['average_response_time']:.2f}s")
        print(f"Requirements Met (95%): {'YES' if summary['requirements_met'] else 'NO'}")
        print("="*60)

        # Exit with appropriate code based on requirements
        if summary['requirements_met']:
            logger.info("✅ EVALUATION PASSED: Accuracy requirement (>95%) met!")
            sys.exit(0)  # Success exit code
        else:
            logger.error("❌ EVALUATION FAILED: Accuracy requirement (>95%) not met!")
            sys.exit(1)  # Failure exit code

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        sys.exit(1)  # Failure exit code


if __name__ == "__main__":
    main()