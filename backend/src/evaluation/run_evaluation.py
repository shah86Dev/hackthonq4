#!/usr/bin/env python3
"""
Automated evaluation script for Book-Embedded RAG Chatbot
This script runs the evaluation framework and validates response accuracy
"""

import argparse
import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.evaluation.evaluator import Evaluator
from tests.data.test_dataset import get_test_dataset, get_all_test_datasets
from src.database import Base
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_evaluation(
    db_url: str,
    dataset_name: str = "basic",
    output_file: str = "evaluation_results.json",
    min_accuracy: float = 0.95
):
    """
    Run the automated evaluation
    """
    logger.info(f"Starting evaluation with dataset: {dataset_name}")
    logger.info(f"Database URL: {db_url}")
    logger.info(f"Minimum required accuracy: {min_accuracy}")

    # Set up database connection
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Get the test dataset
        if dataset_name == "all":
            test_cases = get_all_test_datasets()
        else:
            test_cases = get_test_dataset(dataset_name)

        logger.info(f"Loaded {len(test_cases)} test cases")

        # Initialize the evaluator
        evaluator = Evaluator(db)

        # Run comprehensive evaluation
        logger.info("Starting comprehensive evaluation...")
        start_time = datetime.utcnow()

        report = evaluator.run_comprehensive_evaluation(test_cases)

        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()

        logger.info(f"Evaluation completed in {total_time:.2f} seconds")

        # Save the report
        evaluator.save_evaluation_report(report, output_file)
        logger.info(f"Evaluation report saved to {output_file}")

        # Calculate overall accuracy
        summary = report["summary"]
        overall_accuracy = summary["success_rate"]

        logger.info(f"Overall accuracy: {overall_accuracy:.2%}")
        logger.info(f"Average response time: {summary['average_response_time']:.2f}s")
        logger.info(f"Average similarity: {summary['average_similarity']:.2f}")
        logger.info(f"Average factual accuracy: {summary['average_factual_accuracy']:.2f}")

        # Check if accuracy meets requirements
        if overall_accuracy >= min_accuracy:
            logger.info(f"✅ SUCCESS: Accuracy requirement met ({overall_accuracy:.2%} >= {min_accuracy:.2%})")
            return True
        else:
            logger.error(f"❌ FAILURE: Accuracy requirement not met ({overall_accuracy:.2%} < {min_accuracy:.2%})")
            return False

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser(description="Run automated evaluation for Book-Embedded RAG Chatbot")
    parser.add_argument(
        "--db-url",
        default=os.getenv("NEON_DB_URL", "postgresql://postgres:postgres@localhost/book_rag"),
        help="Database URL for the evaluation"
    )
    parser.add_argument(
        "--dataset",
        choices=["basic", "extended", "performance", "all"],
        default="basic",
        help="Test dataset to use for evaluation"
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.95,
        help="Minimum required accuracy (default: 0.95)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation and return exit code based on accuracy"
    )

    args = parser.parse_args()

    # Run the evaluation
    success = run_evaluation(
        db_url=args.db_url,
        dataset_name=args.dataset,
        output_file=args.output,
        min_accuracy=args.min_accuracy
    )

    # If validate flag is set, exit with appropriate code
    if args.validate:
        if success:
            logger.info("Validation passed - exiting with code 0")
            sys.exit(0)
        else:
            logger.error("Validation failed - exiting with code 1")
            sys.exit(1)

if __name__ == "__main__":
    main()