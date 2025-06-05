import time
from datetime import datetime
from typing import Dict, List, Optional
from statistics import mean, median, stdev
from dataclasses import dataclass
import asyncio
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@dataclass
class LatencyMetrics:
    avg_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    std_dev: float
    min_latency: float
    max_latency: float
    sample_count: int

class PerformanceMonitor:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            "data_processing": [],
            "ui_update": [],
            "simulation_loop": [],
            "orderbook_update": [],
            "market_impact_calc": [],
            "slippage_prediction": [],
            "fee_calculation": []
        }
        self.start_times: Dict[str, float] = {}
        
    @contextmanager
    def measure(self, metric_name: str):
        """Context manager for measuring execution time of a block of code"""
        try:
            start_time = time.perf_counter()
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            self.metrics[metric_name].append(duration_ms)
            
            # Keep only last 1000 measurements for each metric
            if len(self.metrics[metric_name]) > 1000:
                self.metrics[metric_name] = self.metrics[metric_name][-1000:]

    def start_measurement(self, metric_name: str):
        """Start measuring time for a metric"""
        self.start_times[metric_name] = time.perf_counter()

    def end_measurement(self, metric_name: str):
        """End measuring time for a metric"""
        if metric_name in self.start_times:
            duration_ms = (time.perf_counter() - self.start_times[metric_name]) * 1000
            self.metrics[metric_name].append(duration_ms)
            del self.start_times[metric_name]

    def get_metrics(self, metric_name: str) -> Optional[LatencyMetrics]:
        """Calculate statistics for a given metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None

        values = self.metrics[metric_name]
        sorted_values = sorted(values)
        
        return LatencyMetrics(
            avg_latency=mean(values),
            median_latency=median(values),
            p95_latency=sorted_values[int(len(values) * 0.95)],
            p99_latency=sorted_values[int(len(values) * 0.99)],
            std_dev=stdev(values) if len(values) > 1 else 0,
            min_latency=min(values),
            max_latency=max(values),
            sample_count=len(values)
        )

    def get_all_metrics(self) -> Dict[str, LatencyMetrics]:
        """Get metrics for all measured components"""
        return {
            name: self.get_metrics(name)
            for name in self.metrics.keys()
            if self.get_metrics(name) is not None
        }

    def log_metrics(self):
        """Log current metrics to the logger"""
        all_metrics = self.get_all_metrics()
        for metric_name, metrics in all_metrics.items():
            logger.info(f"{metric_name} Latency Metrics:")
            logger.info(f"  Average: {metrics.avg_latency:.2f}ms")
            logger.info(f"  Median: {metrics.median_latency:.2f}ms")
            logger.info(f"  P95: {metrics.p95_latency:.2f}ms")
            logger.info(f"  P99: {metrics.p99_latency:.2f}ms")
            logger.info(f"  Min: {metrics.min_latency:.2f}ms")
            logger.info(f"  Max: {metrics.max_latency:.2f}ms")
            logger.info(f"  StdDev: {metrics.std_dev:.2f}ms")
            logger.info(f"  Samples: {metrics.sample_count}") 