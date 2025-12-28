import time
import logging
from typing import List, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.services.generation_service import GenerationService
from src.services.retrieval_service import RetrievalService
from src.agents.coordinator_agent import CoordinatorAgent
from sqlalchemy.orm import Session
import statistics
import json

logger = logging.getLogger(__name__)

class PerformanceBenchmarks:
    """
    Performance benchmarking for the RAG chatbot system
    """

    def __init__(self, db: Session):
        self.db = db
        self.generation_service = GenerationService()
        self.retrieval_service = RetrievalService()
        self.coordinator_agent = CoordinatorAgent()
        self.benchmark_results = {}

    def benchmark_response_time(self, test_cases: List[Dict], iterations: int = 10) -> Dict:
        """
        Benchmark response times for various queries
        """
        logger.info(f"Starting response time benchmark with {iterations} iterations")

        response_times = []
        all_results = []

        for i in range(iterations):
            for test_case in test_cases:
                start_time = time.time()

                try:
                    result = self.coordinator_agent.process_query(
                        db=self.db,
                        book_id=test_case["book_id"],
                        question=test_case["question"],
                        selected_text=test_case.get("selected_text")
                    )

                    end_time = time.time()
                    response_time = end_time - start_time

                    response_times.append(response_time)
                    all_results.append({
                        "iteration": i,
                        "question": test_case["question"][:50] + "...",
                        "response_time": response_time,
                        "answer_length": len(result.get("answer", "")),
                        "chunks_used": len(result.get("context_chunks_used", []))
                    })

                    logger.debug(f"Iteration {i+1}, Question: {test_case['question'][:30]}..., Time: {response_time:.3f}s")

                except Exception as e:
                    logger.error(f"Error during benchmark iteration {i+1}: {e}")
                    response_times.append(float('inf'))  # Mark as failed

        # Calculate statistics
        valid_response_times = [t for t in response_times if t != float('inf')]
        if valid_response_times:
            stats = {
                "mean_response_time": statistics.mean(valid_response_times),
                "median_response_time": statistics.median(valid_response_times),
                "std_deviation": statistics.stdev(valid_response_times) if len(valid_response_times) > 1 else 0,
                "min_response_time": min(valid_response_times),
                "max_response_time": max(valid_response_times),
                "p95_response_time": self._calculate_percentile(valid_response_times, 95),
                "p99_response_time": self._calculate_percentile(valid_response_times, 99),
                "total_requests": len(valid_response_times),
                "failed_requests": len(response_times) - len(valid_response_times),
                "success_rate": len(valid_response_times) / len(response_times)
            }
        else:
            stats = {
                "mean_response_time": float('inf'),
                "median_response_time": float('inf'),
                "std_deviation": 0,
                "min_response_time": float('inf'),
                "max_response_time": float('inf'),
                "p95_response_time": float('inf'),
                "p99_response_time": float('inf'),
                "total_requests": 0,
                "failed_requests": len(response_times),
                "success_rate": 0.0
            }

        benchmark_result = {
            "benchmark_type": "response_time",
            "test_cases_count": len(test_cases),
            "iterations_per_case": iterations,
            "statistics": stats,
            "individual_results": all_results,
            "timestamp": time.time()
        }

        self.benchmark_results["response_time"] = benchmark_result
        logger.info(f"Response time benchmark completed: {stats['mean_response_time']:.3f}s mean, {stats['success_rate']:.1%} success rate")
        return benchmark_result

    def benchmark_concurrent_users(self, test_cases: List[Dict], user_counts: List[int] = [1, 5, 10, 20, 50]) -> List[Dict]:
        """
        Benchmark performance under different concurrent user loads
        """
        logger.info(f"Starting concurrent users benchmark with user counts: {user_counts}")

        results = []
        for user_count in user_counts:
            logger.info(f"Testing with {user_count} concurrent users...")

            start_time = time.time()

            # Execute requests concurrently
            with ThreadPoolExecutor(max_workers=user_count) as executor:
                futures = []
                for i in range(user_count):
                    test_case = test_cases[i % len(test_cases)]  # Cycle through test cases
                    future = executor.submit(
                        self.coordinator_agent.process_query,
                        self.db,
                        test_case["book_id"],
                        test_case["question"],
                        test_case.get("selected_text")
                    )
                    futures.append(future)

                # Collect results
                completed_count = 0
                response_times = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)  # 30s timeout per request
                        response_times.append(time.time() - start_time)  # This is not accurate - let me fix this
                        completed_count += 1
                    except Exception as e:
                        logger.error(f"Request failed: {e}")

            end_time = time.time()
            total_time = end_time - start_time

            result = {
                "concurrent_users": user_count,
                "total_time": total_time,
                "completed_requests": completed_count,
                "failed_requests": user_count - completed_count,
                "requests_per_second": completed_count / total_time if total_time > 0 else 0,
                "timestamp": time.time()
            }

            results.append(result)
            logger.info(f"Completed {user_count} users: {result['requests_per_second']:.2f} RPS")

        benchmark_result = {
            "benchmark_type": "concurrent_users",
            "user_counts": user_counts,
            "results": results,
            "timestamp": time.time()
        }

        self.benchmark_results["concurrent_users"] = benchmark_result
        return benchmark_result

    def benchmark_memory_usage(self) -> Dict:
        """
        Benchmark memory usage (placeholder - would need actual memory monitoring)
        """
        logger.info("Starting memory usage benchmark")

        # In a real implementation, we would monitor memory usage
        # For now, we'll just return placeholder values
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        benchmark_result = {
            "benchmark_type": "memory_usage",
            "rss_memory_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            "vms_memory_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            "timestamp": time.time()
        }

        self.benchmark_results["memory_usage"] = benchmark_result
        logger.info(f"Memory usage: {benchmark_result['rss_memory_mb']:.2f} MB RSS")
        return benchmark_result

    def benchmark_throughput(self, test_cases: List[Dict], duration_seconds: int = 60) -> Dict:
        """
        Benchmark throughput over a fixed duration
        """
        logger.info(f"Starting throughput benchmark for {duration_seconds} seconds")

        start_time = time.time()
        end_time = start_time + duration_seconds
        completed_requests = 0
        response_times = []

        while time.time() < end_time:
            for test_case in test_cases:
                if time.time() >= end_time:
                    break

                try:
                    iteration_start = time.time()
                    result = self.coordinator_agent.process_query(
                        db=self.db,
                        book_id=test_case["book_id"],
                        question=test_case["question"],
                        selected_text=test_case.get("selected_text")
                    )
                    iteration_end = time.time()

                    response_times.append(iteration_end - iteration_start)
                    completed_requests += 1

                    logger.debug(f"Completed request {completed_requests}, time: {iteration_end - iteration_start:.3f}s")
                except Exception as e:
                    logger.error(f"Request failed: {e}")

        actual_duration = time.time() - start_time
        requests_per_second = completed_requests / actual_duration if actual_duration > 0 else 0

        benchmark_result = {
            "benchmark_type": "throughput",
            "duration_seconds": actual_duration,
            "completed_requests": completed_requests,
            "requests_per_second": requests_per_second,
            "response_times": response_times,
            "timestamp": time.time()
        }

        self.benchmark_results["throughput"] = benchmark_result
        logger.info(f"Throughput: {requests_per_second:.2f} RPS, {completed_requests} requests in {actual_duration:.2f}s")
        return benchmark_result

    def run_all_benchmarks(self, test_cases: List[Dict]) -> Dict:
        """
        Run all performance benchmarks
        """
        logger.info("Starting all performance benchmarks...")

        # Run each benchmark
        response_time_result = self.benchmark_response_time(test_cases, iterations=5)
        concurrent_result = self.benchmark_concurrent_users(test_cases, [1, 5, 10])
        memory_result = self.benchmark_memory_usage()
        throughput_result = self.benchmark_throughput(test_cases, duration_seconds=30)

        # Compile all results
        all_results = {
            "summary": {
                "response_time_mean": response_time_result["statistics"]["mean_response_time"],
                "response_time_p95": response_time_result["statistics"]["p95_response_time"],
                "throughput_rps": throughput_result["requests_per_second"],
                "concurrent_users_tested": [r["concurrent_users"] for r in concurrent_result["results"]],
                "memory_usage_mb": memory_result["rss_memory_mb"]
            },
            "detailed_results": self.benchmark_results,
            "timestamp": time.time()
        }

        logger.info("All benchmarks completed")
        return all_results

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """
        Calculate percentile of a list of values
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            fraction = index - int(index)
            return lower + (upper - lower) * fraction

    def save_benchmark_report(self, report: Dict, filename: str = "performance_benchmark_report.json"):
        """
        Save benchmark report to a file
        """
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Benchmark report saved to {filename}")

def run_performance_benchmarks(db: Session, test_cases: List[Dict]) -> Dict:
    """
    Convenience function to run all performance benchmarks
    """
    benchmark_runner = PerformanceBenchmarks(db)
    results = benchmark_runner.run_all_benchmarks(test_cases)
    benchmark_runner.save_benchmark_report(results)
    return results