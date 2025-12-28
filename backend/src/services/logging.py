import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
import os

class EvaluationLogger:
    """
    Comprehensive logging service for evaluation metrics and system performance
    """

    def __init__(self, log_file: str = "evaluation.log", max_bytes: int = 10*1024*1024, backup_count: int = 5):
        """
        Initialize the evaluation logger
        """
        self.logger = logging.getLogger('evaluation')
        self.logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create file handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_evaluation_start(self, evaluation_id: str, test_count: int, dataset_name: str = "unknown"):
        """
        Log the start of an evaluation run
        """
        message = f"Evaluation started - ID: {evaluation_id}, Tests: {test_count}, Dataset: {dataset_name}"
        self.logger.info(message)

    def log_evaluation_result(self, evaluation_id: str, test_case: Dict, result: Dict):
        """
        Log individual evaluation results
        """
        log_data = {
            "evaluation_id": evaluation_id,
            "question": test_case.get("question", "")[:100] + "..." if len(test_case.get("question", "")) > 100 else test_case.get("question", ""),
            "book_id": test_case.get("book_id"),
            "selected_text_present": bool(test_case.get("selected_text")),
            "response_time": result.get("response_time", 0),
            "similarity_score": result["metrics"].get("similarity_score", 0),
            "factual_accuracy": result["metrics"].get("factual_accuracy", 0),
            "overall_quality": result["metrics"].get("overall_quality", 0),
            "timestamp": datetime.utcnow().isoformat()
        }

        message = f"Evaluation result - {json.dumps(log_data)}"
        self.logger.info(message)

    def log_evaluation_summary(self, evaluation_id: str, summary: Dict):
        """
        Log evaluation summary
        """
        log_data = {
            "evaluation_id": evaluation_id,
            "total_tests": summary.get("total_tests", 0),
            "successful_tests": summary.get("successful_tests", 0),
            "success_rate": summary.get("success_rate", 0),
            "average_response_time": summary.get("average_response_time", 0),
            "average_similarity": summary.get("average_similarity", 0),
            "average_factual_accuracy": summary.get("average_factual_accuracy", 0),
            "timestamp": datetime.utcnow().isoformat()
        }

        message = f"Evaluation summary - {json.dumps(log_data)}"
        self.logger.info(message)

    def log_error(self, evaluation_id: str, error: Exception, context: str = ""):
        """
        Log evaluation errors
        """
        log_data = {
            "evaluation_id": evaluation_id,
            "error": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }

        message = f"Evaluation error - {json.dumps(log_data)}"
        self.logger.error(message)

    def log_performance_metrics(self, evaluation_id: str, metrics: Dict):
        """
        Log detailed performance metrics
        """
        log_data = {
            "evaluation_id": evaluation_id,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }

        message = f"Performance metrics - {json.dumps(log_data)}"
        self.logger.info(message)

    def log_system_metrics(self, evaluation_id: str, system_metrics: Dict):
        """
        Log system-level metrics during evaluation
        """
        log_data = {
            "evaluation_id": evaluation_id,
            "system_metrics": system_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }

        message = f"System metrics - {json.dumps(log_data)}"
        self.logger.info(message)

class SystemLogger:
    """
    System-level logging for the RAG chatbot
    """

    def __init__(self, log_file: str = "system.log", max_bytes: int = 10*1024*1024, backup_count: int = 5):
        """
        Initialize the system logger
        """
        self.logger = logging.getLogger('system')
        self.logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create file handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_api_request(self, endpoint: str, method: str, user_id: str = None, session_id: str = None):
        """
        Log API requests
        """
        log_data = {
            "endpoint": endpoint,
            "method": method,
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        message = f"API request - {json.dumps(log_data)}"
        self.logger.info(message)

    def log_api_response(self, endpoint: str, status_code: int, response_time: float, user_id: str = None):
        """
        Log API responses
        """
        log_data = {
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time": response_time,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        message = f"API response - {json.dumps(log_data)}"
        self.logger.info(message)

    def log_ingestion_start(self, book_id: str, file_name: str, file_size: int):
        """
        Log the start of a book ingestion process
        """
        log_data = {
            "book_id": book_id,
            "file_name": file_name,
            "file_size": file_size,
            "timestamp": datetime.utcnow().isoformat()
        }

        message = f"Ingestion started - {json.dumps(log_data)}"
        self.logger.info(message)

    def log_ingestion_complete(self, book_id: str, chunks_created: int, processing_time: float):
        """
        Log the completion of a book ingestion process
        """
        log_data = {
            "book_id": book_id,
            "chunks_created": chunks_created,
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }

        message = f"Ingestion completed - {json.dumps(log_data)}"
        self.logger.info(message)

    def log_query_processing(self, query_id: str, book_id: str, query_text: str, mode: str = "full-book"):
        """
        Log query processing events
        """
        log_data = {
            "query_id": query_id,
            "book_id": book_id,
            "query_text": query_text[:100] + "..." if len(query_text) > 100 else query_text,
            "mode": mode,
            "timestamp": datetime.utcnow().isoformat()
        }

        message = f"Query processing - {json.dumps(log_data)}"
        self.logger.info(message)

    def log_error_event(self, error_type: str, error_message: str, context: str = ""):
        """
        Log system errors
        """
        log_data = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }

        message = f"System error - {json.dumps(log_data)}"
        self.logger.error(message)

# Global loggers
evaluation_logger = EvaluationLogger()
system_logger = SystemLogger()