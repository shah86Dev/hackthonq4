import json
import os
from datetime import datetime
from typing import Dict, List
import logging
from jinja2 import Template
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generate evaluation reports and dashboards for the RAG chatbot
    """

    def __init__(self):
        self.reports_dir = "reports"
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)

    def generate_evaluation_report(self, evaluation_results: Dict) -> str:
        """
        Generate a comprehensive evaluation report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_data = {
            "timestamp": timestamp,
            "summary": evaluation_results.get("summary", {}),
            "detailed_analysis": evaluation_results.get("detailed_analysis", {}),
            "metrics": self._extract_metrics(evaluation_results)
        }

        # Create HTML report
        html_content = self._generate_html_report(report_data)
        report_filename = os.path.join(self.reports_dir, f"evaluation_report_{int(datetime.now().timestamp())}.html")

        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Evaluation report generated: {report_filename}")
        return report_filename

    def generate_performance_report(self, benchmark_results: Dict) -> str:
        """
        Generate a performance benchmark report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_data = {
            "timestamp": timestamp,
            "summary": benchmark_results.get("summary", {}),
            "detailed_results": benchmark_results.get("detailed_results", {}),
            "charts": self._generate_performance_charts(benchmark_results)
        }

        # Create HTML report
        html_content = self._generate_performance_html_report(report_data)
        report_filename = os.path.join(self.reports_dir, f"performance_report_{int(datetime.now().timestamp())}.html")

        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Performance report generated: {report_filename}")
        return report_filename

    def _extract_metrics(self, evaluation_results: Dict) -> Dict:
        """
        Extract key metrics from evaluation results
        """
        summary = evaluation_results.get("summary", {})
        detailed = evaluation_results.get("detailed_analysis", {})

        metrics = {
            "total_tests": summary.get("total_tests", 0),
            "successful_tests": summary.get("successful_tests", 0),
            "success_rate": summary.get("success_rate", 0),
            "average_response_time": summary.get("average_response_time", 0),
            "average_similarity": summary.get("average_similarity", 0),
            "average_factual_accuracy": summary.get("average_factual_accuracy", 0),
            "high_similarity_count": detailed.get("high_similarity_count", 0),
            "low_similarity_count": detailed.get("low_similarity_count", 0),
            "response_time_percentiles": detailed.get("response_time_percentiles", {}),
            "similarity_percentiles": detailed.get("similarity_percentiles", {})
        }

        return metrics

    def _generate_html_report(self, report_data: Dict) -> str:
        """
        Generate HTML report using a template
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Book-Embedded RAG Chatbot - Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2 { color: #2c3e50; }
                .header { background-color: #3498db; color: white; padding: 20px; border-radius: 5px; }
                .summary { background-color: #ecf0f1; padding: 20px; margin: 20px 0; border-radius: 5px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .metric-card { background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .metric-value { font-size: 2em; font-weight: bold; color: #27ae60; }
                .metric-title { color: #7f8c8d; font-size: 0.9em; }
                .chart-container { margin: 20px 0; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #3498db; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Book-Embedded RAG Chatbot - Evaluation Report</h1>
                <p>Generated on: {{ timestamp }}</p>
            </div>

            <div class="summary">
                <h2>Evaluation Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(metrics.success_rate * 100) }}%</div>
                        <div class="metric-title">Success Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(metrics.average_response_time) }}s</div>
                        <div class="metric-title">Avg. Response Time</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(metrics.average_similarity * 100) }}%</div>
                        <div class="metric-title">Avg. Similarity</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(metrics.average_factual_accuracy * 100) }}%</div>
                        <div class="metric-title">Avg. Accuracy</div>
                    </div>
                </div>
            </div>

            <div>
                <h2>Detailed Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Tests</td>
                        <td>{{ metrics.total_tests }}</td>
                    </tr>
                    <tr>
                        <td>Successful Tests</td>
                        <td>{{ metrics.successful_tests }}</td>
                    </tr>
                    <tr>
                        <td>High Similarity Responses (>=0.8)</td>
                        <td>{{ metrics.high_similarity_count }}</td>
                    </tr>
                    <tr>
                        <td>Low Similarity Responses (<0.5)</td>
                        <td>{{ metrics.low_similarity_count }}</td>
                    </tr>
                    <tr>
                        <td>Response Time P95</td>
                        <td>{{ "%.3f"|format(metrics.response_time_percentiles.get(95, 0)) }}s</td>
                    </tr>
                    <tr>
                        <td>Similarity P95</td>
                        <td>{{ "%.2f"|format(metrics.similarity_percentiles.get(95, 0) * 100) }}%</td>
                    </tr>
                </table>
            </div>

            <div>
                <h2>Test Results</h2>
                <table>
                    <tr>
                        <th>Question</th>
                        <th>Response Time</th>
                        <th>Similarity</th>
                        <th>Accuracy</th>
                    </tr>
                    {% for result in summary.results[:10] %}
                    <tr>
                        <td>{{ result.question[:50] + "..." if result.question|length > 50 else result.question }}</td>
                        <td>{{ "%.3f"|format(result.response_time) }}s</td>
                        <td>{{ "%.2f"|format(result.metrics.similarity_score * 100) }}%</td>
                        <td>{{ "%.2f"|format(result.metrics.factual_accuracy * 100) }}%</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </body>
        </html>
        """

        template = Template(html_template)
        return template.render(**report_data)

    def _generate_performance_html_report(self, report_data: Dict) -> str:
        """
        Generate HTML report for performance benchmarks
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Book-Embedded RAG Chatbot - Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2 { color: #2c3e50; }
                .header { background-color: #9b59b6; color: white; padding: 20px; border-radius: 5px; }
                .summary { background-color: #ecf0f1; padding: 20px; margin: 20px 0; border-radius: 5px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .metric-card { background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .metric-value { font-size: 2em; font-weight: bold; color: #8e44ad; }
                .metric-title { color: #7f8c8d; font-size: 0.9em; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #9b59b6; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Book-Embedded RAG Chatbot - Performance Report</h1>
                <p>Generated on: {{ timestamp }}</p>
            </div>

            <div class="summary">
                <h2>Performance Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(summary.response_time_mean) }}s</div>
                        <div class="metric-title">Mean Response Time</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(summary.response_time_p95) }}s</div>
                        <div class="metric-title">P95 Response Time</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(summary.throughput_rps) }}</div>
                        <div class="metric-title">Requests/Second</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.1f"|format(summary.memory_usage_mb) }}MB</div>
                        <div class="metric-title">Memory Usage</div>
                    </div>
                </div>
            </div>

            <div>
                <h2>Throughput Results</h2>
                <table>
                    <tr>
                        <th>Duration (s)</th>
                        <th>Requests Completed</th>
                        <th>Requests/Second</th>
                    </tr>
                    <tr>
                        <td>{{ "%.2f"|format(detailed_results.throughput.duration_seconds) }}</td>
                        <td>{{ detailed_results.throughput.completed_requests }}</td>
                        <td>{{ "%.2f"|format(detailed_results.throughput.requests_per_second) }}</td>
                    </tr>
                </table>
            </div>

            <div>
                <h2>Concurrent Users Performance</h2>
                <table>
                    <tr>
                        <th>Concurrent Users</th>
                        <th>Requests/Second</th>
                        <th>Success Rate</th>
                        <th>Failed Requests</th>
                    </tr>
                    {% for result in detailed_results.concurrent_users.results %}
                    <tr>
                        <td>{{ result.concurrent_users }}</td>
                        <td>{{ "%.2f"|format(result.requests_per_second) }}</td>
                        <td>{{ "%.1f"|format(result.completed_requests / (result.completed_requests + result.failed_requests) * 100) }}%</td>
                        <td>{{ result.failed_requests }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </body>
        </html>
        """

        template = Template(html_template)
        return template.render(**report_data)

    def _generate_performance_charts(self, benchmark_results: Dict) -> Dict:
        """
        Generate performance charts
        """
        charts = {}

        # Response time chart
        if 'response_time' in benchmark_results.get('detailed_results', {}):
            response_time_data = benchmark_results['detailed_results']['response_time']
            stats = response_time_data.get('statistics', {})

            if stats:
                plt.figure(figsize=(10, 6))
                plt.bar(['Mean', 'Median', 'P95'], [
                    stats.get('mean_response_time', 0),
                    stats.get('median_response_time', 0),
                    stats.get('p95_response_time', 0)
                ])
                plt.title('Response Time Statistics (seconds)')
                plt.ylabel('Time (s)')

                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png')
                img_buffer.seek(0)
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()

                charts['response_time'] = f"data:image/png;base64,{img_str}"

        return charts

    def generate_dashboard(self, evaluation_results: Dict, benchmark_results: Dict) -> str:
        """
        Generate a comprehensive dashboard combining evaluation and performance data
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        dashboard_data = {
            "timestamp": timestamp,
            "evaluation_summary": evaluation_results.get("summary", {}),
            "performance_summary": benchmark_results.get("summary", {}) if benchmark_results else {},
            "evaluation_metrics": self._extract_metrics(evaluation_results),
            "overall_status": self._determine_overall_status(evaluation_results, benchmark_results)
        }

        html_content = self._generate_dashboard_html(dashboard_data)
        dashboard_filename = os.path.join(self.reports_dir, f"dashboard_{int(datetime.now().timestamp())}.html")

        with open(dashboard_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Dashboard generated: {dashboard_filename}")
        return dashboard_filename

    def _determine_overall_status(self, evaluation_results: Dict, benchmark_results: Dict) -> str:
        """
        Determine overall system status based on evaluation and performance results
        """
        eval_summary = evaluation_results.get("summary", {})
        perf_summary = benchmark_results.get("summary", {}) if benchmark_results else {}

        success_rate = eval_summary.get("success_rate", 0)
        avg_response_time = eval_summary.get("average_response_time", float('inf'))
        throughput = perf_summary.get("throughput_rps", 0)

        # Define thresholds
        success_rate_threshold = 0.90  # 90% success rate
        response_time_threshold = 2.0  # 2 seconds max average
        throughput_threshold = 1.0  # 1 request per second minimum

        if (success_rate >= success_rate_threshold and
            avg_response_time <= response_time_threshold and
            throughput >= throughput_threshold):
            return "healthy"
        elif (success_rate >= success_rate_threshold * 0.8 and  # 72% threshold
              avg_response_time <= response_time_threshold * 2 and  # 4 seconds max
              throughput >= throughput_threshold * 0.5):  # 0.5 RPS minimum
            return "warning"
        else:
            return "critical"

    def _generate_dashboard_html(self, dashboard_data: Dict) -> str:
        """
        Generate HTML dashboard
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Book-Embedded RAG Chatbot - Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }
                .header { background-color: #2c3e50; color: white; padding: 20px; }
                .status-bar { padding: 10px 20px; font-weight: bold; }
                .status-healthy { background-color: #27ae60; color: white; }
                .status-warning { background-color: #f39c12; color: white; }
                .status-critical { background-color: #e74c3c; color: white; }
                .container { max-width: 1200px; margin: 20px auto; padding: 20px; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .card { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .metric-value { font-size: 2.5em; font-weight: bold; margin: 10px 0; }
                .metric-title { color: #7f8c8d; font-size: 1em; }
                .section-title { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin: 20px 0 10px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Book-Embedded RAG Chatbot - System Dashboard</h1>
                <p>Generated on: {{ timestamp }}</p>
            </div>

            <div class="status-bar status-{{ overall_status }}">
                System Status: {{ overall_status|title }}
            </div>

            <div class="container">
                <div class="grid">
                    <div class="card">
                        <div class="metric-value">{{ "%.2f"|format(evaluation_metrics.success_rate * 100) }}%</div>
                        <div class="metric-title">Evaluation Success Rate</div>
                    </div>
                    <div class="card">
                        <div class="metric-value">{{ "%.2f"|format(evaluation_metrics.average_response_time) }}s</div>
                        <div class="metric-title">Avg. Response Time</div>
                    </div>
                    <div class="card">
                        <div class="metric-value">{{ "%.2f"|format(performance_summary.throughput_rps) }}</div>
                        <div class="metric-title">Throughput (RPS)</div>
                    </div>
                    <div class="card">
                        <div class="metric-value">{{ "%.1f"|format(performance_summary.memory_usage_mb) }}MB</div>
                        <div class="metric-title">Memory Usage</div>
                    </div>
                </div>

                <h2 class="section-title">Evaluation Metrics</h2>
                <div class="grid">
                    <div class="card">
                        <div class="metric-value">{{ evaluation_metrics.total_tests }}</div>
                        <div class="metric-title">Total Tests</div>
                    </div>
                    <div class="card">
                        <div class="metric-value">{{ "%.2f"|format(evaluation_metrics.average_similarity * 100) }}%</div>
                        <div class="metric-title">Avg. Similarity</div>
                    </div>
                    <div class="card">
                        <div class="metric-value">{{ "%.2f"|format(evaluation_metrics.average_factual_accuracy * 100) }}%</div>
                        <div class="metric-title">Avg. Accuracy</div>
                    </div>
                </div>

                <h2 class="section-title">Performance Metrics</h2>
                <div class="grid">
                    <div class="card">
                        <div class="metric-value">{{ "%.2f"|format(performance_summary.response_time_p95) }}s</div>
                        <div class="metric-title">P95 Response Time</div>
                    </div>
                    <div class="card">
                        <div class="metric-value">{{ performance_summary.concurrent_users_tested|join(', ') }}</div>
                        <div class="metric-title">Concurrent Users Tested</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        template = Template(html_template)
        return template.render(**dashboard_data)

def generate_full_report(evaluation_results: Dict, benchmark_results: Dict = None) -> Dict:
    """
    Generate all reports for a complete evaluation
    """
    generator = ReportGenerator()

    # Generate evaluation report
    eval_report = generator.generate_evaluation_report(evaluation_results)

    # Generate performance report if available
    perf_report = None
    if benchmark_results:
        perf_report = generator.generate_performance_report(benchmark_results)

    # Generate dashboard
    dashboard = generator.generate_dashboard(evaluation_results, benchmark_results or {})

    return {
        "evaluation_report": eval_report,
        "performance_report": perf_report,
        "dashboard": dashboard,
        "timestamp": datetime.now().isoformat()
    }