"""Deterministic failure probes and structured diagnostic reports."""

from .diagnoser import DiagnosisThresholds, build_failure_report, summarize_horizon_curve

__all__ = ["DiagnosisThresholds", "build_failure_report", "summarize_horizon_curve"]
