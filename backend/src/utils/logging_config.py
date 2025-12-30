import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
from datetime import datetime
import json
from typing import Dict, Any

# Optional import for pythonjsonlogger
try:
    from pythonjsonlogger import jsonlogger
    PYTHONJSONLOGGER_AVAILABLE = True
except ImportError:
    PYTHONJSONLOGGER_AVAILABLE = False
    # Create a simple fallback formatter if pythonjsonlogger is not available
    class SimpleJsonFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
            }
            if hasattr(record, 'funcName'):
                log_entry['function'] = record.funcName
            if hasattr(record, 'lineno'):
                log_entry['line'] = record.lineno
            return json.dumps(log_entry)

    jsonlogger = None

# Optional import for structlog
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

# Set up structlog configuration
def setup_structured_logging():
    """Set up structured logging for the application"""
    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging if structlog is not available
        return get_logger("structlog_fallback")

    # Configure processor chain for structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()


def setup_logging(log_level: str = "INFO", log_file: str = "app.log"):
    """
    Set up comprehensive logging for the application
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create formatters
    if PYTHONJSONLOGGER_AVAILABLE:
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            rename_fields={'asctime': 'timestamp', 'name': 'logger', 'levelname': 'level'}
        )
    else:
        json_formatter = SimpleJsonFormatter()

    standard_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(standard_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation based on size
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(json_formatter)
    root_logger.addHandler(file_handler)

    # Error file handler (only errors and above)
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, "error.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)
    root_logger.addHandler(error_handler)

    # Audit log handler for security-related events
    audit_handler = RotatingFileHandler(
        os.path.join(log_dir, "audit.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(json_formatter)
    audit_logger = logging.getLogger("audit")
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)

    # Performance log handler
    perf_handler = RotatingFileHandler(
        os.path.join(log_dir, "performance.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(json_formatter)
    perf_logger = logging.getLogger("performance")
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)

    return root_logger


def get_logger(name: str = None):
    """
    Get a configured logger instance
    """
    if name:
        return logging.getLogger(name)
    else:
        return logging.getLogger()


def log_api_call(
    endpoint: str,
    method: str,
    user_id: str = None,
    ip_address: str = None,
    response_time: float = None,
    status_code: int = None
):
    """
    Log API call details
    """
    logger = get_logger("api")
    logger.info(
        "API Call",
        extra={
            "endpoint": endpoint,
            "method": method,
            "user_id": user_id,
            "ip_address": ip_address,
            "response_time": response_time,
            "status_code": status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def log_query_execution(
    query_id: str,
    book_id: str,
    question: str,
    response_time: float,
    tokens_used: int,
    confidence_score: float,
    user_id: str = None
):
    """
    Log query execution details
    """
    logger = get_logger("query")
    logger.info(
        "Query Executed",
        extra={
            "query_id": query_id,
            "book_id": book_id,
            "question": question[:100] + "..." if len(question) > 100 else question,  # Truncate long questions
            "response_time": response_time,
            "tokens_used": tokens_used,
            "confidence_score": confidence_score,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def log_ingestion_event(
    book_id: str,
    title: str,
    file_type: str,
    file_size: int,
    chunks_created: int,
    processing_time: float,
    status: str = "completed"
):
    """
    Log book ingestion events
    """
    logger = get_logger("ingestion")
    logger.info(
        "Book Ingestion",
        extra={
            "book_id": book_id,
            "title": title,
            "file_type": file_type,
            "file_size": file_size,
            "chunks_created": chunks_created,
            "processing_time": processing_time,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def log_error_event(
    error_type: str,
    error_message: str,
    endpoint: str = None,
    user_id: str = None,
    traceback_info: str = None
):
    """
    Log error events with details
    """
    logger = get_logger("error")
    logger.error(
        "Error Occurred",
        extra={
            "error_type": error_type,
            "error_message": error_message,
            "endpoint": endpoint,
            "user_id": user_id,
            "traceback": traceback_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def log_performance_metric(
    metric_name: str,
    value: float,
    unit: str = None,
    context: Dict[str, Any] = None
):
    """
    Log performance metrics
    """
    logger = get_logger("performance")
    extra_data = {
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
        "timestamp": datetime.utcnow().isoformat()
    }

    if context:
        extra_data.update(context)

    logger.info(
        "Performance Metric",
        extra=extra_data
    )


def log_audit_event(
    event_type: str,
    user_id: str,
    resource: str,
    action: str,
    success: bool = True,
    details: Dict[str, Any] = None
):
    """
    Log audit events for security and compliance
    """
    logger = get_logger("audit")
    extra_data = {
        "event_type": event_type,
        "user_id": user_id,
        "resource": resource,
        "action": action,
        "success": success,
        "timestamp": datetime.utcnow().isoformat()
    }

    if details:
        extra_data.update(details)

    logger.info(
        "Audit Event",
        extra=extra_data
    )


class PerformanceTimer:
    """
    Context manager for measuring execution time
    """
    def __init__(self, operation_name: str, logger_name: str = "performance"):
        self.operation_name = operation_name
        self.logger_name = logger_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.utcnow()
        duration = (self.end_time - self.start_time).total_seconds()

        log_performance_metric(
            metric_name=f"{self.operation_name}_execution_time",
            value=duration,
            unit="seconds",
            context={
                "operation": self.operation_name,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat()
            }
        )


def measure_performance(operation_name: str):
    """
    Decorator to measure performance of functions
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceTimer(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Initialize logging when module is imported
if __name__ != '__main__':
    setup_logging()
    logger = get_logger(__name__)
    logger.info("Logging configuration initialized")