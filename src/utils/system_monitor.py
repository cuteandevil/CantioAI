"""
System Monitor for CantioAI Complete System
Provides health checks, performance monitoring, and system status
"""

import psutil
import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitors system health, performance, and resource usage"""

    def __init__(self):
        self.start_time = time.time()
        self.metrics_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)

        # Thresholds for alerts
        self.thresholds = {
            "cpu_usage_percent": 80.0,
            "memory_usage_percent": 85.0,
            "disk_usage_percent": 90.0,
            "gpu_usage_percent": 90.0,
            "response_time_ms": 1000.0,
            "error_rate_percent": 5.0
        }

        # Start monitoring thread
        self.start_monitoring()

    def start_monitoring(self) -> None:
        """Start the background monitoring thread"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("[MONITOR] System monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the background monitoring thread"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("[MONITOR] System monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self.collect_system_metrics()
                self.metrics_history.append(metrics)

                # Keep only last 1000 entries to prevent memory growth
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]

                # Check for alerts
                self._check_alerts(metrics)

                # Sleep until next collection
                time.sleep(15)  # Collect metrics every 15 seconds

            except Exception as e:
                self.logger.error(f"[ERROR] Error in monitoring loop: {e}")
                time.sleep(5)  # Short sleep on error

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)

            # GPU usage (if available)
            gpu_percent = 0.0
            gpu_memory_used = 0
            gpu_memory_total = 0
            try:
                import torch
                if torch.cuda.is_available():
                    # This is a simplified GPU check
                    gpu_percent = 0.0  # Would need nvidia-ml-py or similar for actual usage
                    gpu_memory_used = 0
                    gpu_memory_total = 0
            except ImportError:
                pass

            # Process-specific metrics
            process = psutil.Process()
            process_cpu = process.cpu_percent()
            process_memory = process.memory_info().rss / (1024**2)  # MB

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "cpu": {
                    "usage_percent": cpu_percent,
                    "process_usage_percent": process_cpu,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "usage_percent": memory_percent,
                    "used_gb": round(memory_used_gb, 2),
                    "total_gb": round(memory_total_gb, 2),
                    "process_used_mb": round(process_memory, 2)
                },
                "disk": {
                    "usage_percent": disk_percent,
                    "used_gb": round(disk_used_gb, 2),
                    "total_gb": round(disk_total_gb, 2)
                },
                "gpu": {
                    "usage_percent": gpu_percent,
                    "memory_used_mb": round(gpu_memory_used / 1024, 2) if gpu_memory_used > 0 else 0,
                    "memory_total_mb": round(gpu_memory_total / 1024, 2) if gpu_memory_total > 0 else 0,
                    "available": gpu_memory_total > 0
                },
                "process": {
                    "pid": process.pid,
                    "name": process.name(),
                    "status": process.status(),
                    "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
                }
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def _check_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check metrics against thresholds and generate alerts"""
        try:
            alerts = []

            # CPU alert
            if metrics.get("cpu", {}).get("usage_percent", 0) > self.thresholds["cpu_usage_percent"]:
                alerts.append({
                    "type": "cpu_high",
                    "message": f"CPU usage high: {metrics['cpu']['usage_percent']:.1f}%",
                    "value": metrics["cpu"]["usage_percent"],
                    "threshold": self.thresholds["cpu_usage_percent"],
                    "timestamp": metrics["timestamp"]
                })

            # Memory alert
            if metrics.get("memory", {}).get("usage_percent", 0) > self.thresholds["memory_usage_percent"]:
                alerts.append({
                    "type": "memory_high",
                    "message": f"Memory usage high: {metrics['memory']['usage_percent']:.1f}%",
                    "value": metrics["memory"]["usage_percent"],
                    "threshold": self.thresholds["memory_usage_percent"],
                    "timestamp": metrics["timestamp"]
                })

            # Disk alert
            if metrics.get("disk", {}).get("usage_percent", 0) > self.thresholds["disk_usage_percent"]:
                alerts.append({
                    "type": "disk_high",
                    "message": f"Disk usage high: {metrics['disk']['usage_percent']:.1f}%",
                    "value": metrics["disk"]["usage_percent"],
                    "threshold": self.thresholds["disk_usage_percent"],
                    "timestamp": metrics["timestamp"]
                })

            # Add alerts to history
            for alert in alerts:
                self.alerts.append(alert)
                self.logger.warning(f"[ALERT] System alert: {alert['message']}")

            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]

        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")

    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "[NO DATA] No metrics collected yet"}

        latest = self.metrics_history[-1]
        return {
            "status": "[OK] healthy",
            "timestamp": latest["timestamp"],
            "uptime_hours": round((time.time() - self.start_time) / 3600, 2),
            "metrics": latest,
            "alerts_count": len(self.alerts),
            "recent_alerts": self.alerts[-5:] if self.alerts else []
        }

    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"].replace('Z', '+00:00')).timestamp() > cutoff_time
        ]

        if not recent_metrics:
            return {"status": "no_data", "hours": hours}

        # Calculate averages
        cpu_values = [m.get("cpu", {}).get("usage_percent", 0) for m in recent_metrics]
        memory_values = [m.get("memory", {}).get("usage_percent", 0) for m in recent_metrics]
        disk_values = [m.get("disk", {}).get("usage_percent", 0) for m in recent_metrics]

        return {
            "status": "available",
            "hours": hours,
            "samples": len(recent_metrics),
            "averages": {
                "cpu_usage_percent": round(sum(cpu_values) / len(cpu_values), 2) if cpu_values else 0,
                "memory_usage_percent": round(sum(memory_values) / len(memory_values), 2) if memory_values else 0,
                "disk_usage_percent": round(sum(disk_values) / len(disk_values), 2) if disk_values else 0
            },
            "peaks": {
                "cpu_usage_percent": round(max(cpu_values), 2) if cpu_values else 0,
                "memory_usage_percent": round(max(memory_values), 2) if memory_values else 0,
                "disk_usage_percent": round(max(disk_values), 2) if disk_values else 0
            },
            "mins": {
                "cpu_usage_percent": round(min(cpu_values), 2) if cpu_values else 0,
                "memory_usage_percent": round(min(memory_values), 2) if memory_values else 0,
                "disk_usage_percent": round(min(disk_values), 2) if disk_values else 0
            }
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        status = self.get_current_status()

        # Determine overall health
        is_healthy = True
        issues = []

        if "error" in status:
            is_healthy = False
            issues.append("Metrics collection error")

        # Check recent alerts for critical issues
        recent_alerts = status.get("recent_alerts", [])
        critical_alerts = [
            alert for alert in recent_alerts
            if alert.get("type") in ["cpu_high", "memory_high", "disk_high"]
        ]

        if critical_alerts:
            is_healthy = False
            issues.append(f"{len(critical_alerts)} critical resource alerts")

        return {
            "healthy": is_healthy,
            "status": "healthy" if is_healthy else "degraded",
            "issues": issues,
            "timestamp": status.get("timestamp"),
            "uptime_hours": status.get("uptime_hours", 0),
            "metrics_available": "error" not in status
        }


# Global monitor instance
_system_monitor = None


def get_system_monitor() -> SystemMonitor:
    """Get the system monitor (singleton pattern)"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor


def start_system_monitoring() -> None:
    """Start system monitoring"""
    monitor = get_system_monitor()
    monitor.start_monitoring()


def stop_system_monitoring() -> None:
    """Stop system monitoring"""
    monitor = get_system_monitor()
    monitor.stop_monitoring()


def get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    monitor = get_system_monitor()
    return monitor.get_current_status()


def get_system_health() -> Dict[str, Any]:
    """Get system health status"""
    monitor = get_system_monitor()
    return monitor.get_health_status()


if __name__ == "__main__":
    # Test the system monitor
    logging.basicConfig(level=logging.INFO)

    try:
        monitor = SystemMonitor()

        # Collect a few metrics
        for i in range(3):
            metrics = monitor.collect_system_metrics()
            print(f"Metrics collection {i+1}:")
            print(f"  CPU: {metrics.get('cpu', {}).get('usage_percent', 0):.1f}%")
            print(f"  Memory: {metrics.get('memory', {}).get('usage_percent', 0):.1f}%")
            print(f"  Disk: {metrics.get('disk', {}).get('usage_percent', 0):.1f}%")
            time.sleep(2)

        # Get status
        status = monitor.get_current_status()
        print(f"\nCurrent status: {status.get('status')}")

        # Get health
        health = monitor.get_health_status()
        print(f"Health status: {health.get('status')} ({'healthy' if health.get('healthy') else 'unhealthy'})")

        # Stop monitoring
        monitor.stop_monitoring()
        print("[SUCCESS] System monitor test completed")

    except Exception as e:
        print(f"[FAIL] System monitor test failed: {e}")