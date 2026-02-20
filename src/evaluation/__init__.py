"""Evaluation module with Anti-Bias Framework."""

from evaluation.antibias_validator import (
    PerformanceMetrics,
    compute_metrics,
    CPCVResult,
    CPCVEvaluator,
    PermutationResult,
    PermutationTest,
    DSRResult,
    DeflatedSharpeRatio,
    MTRLResult,
    MinTrackRecordLength,
    ValidationReport,
    BacktestValidator,
)

__all__ = [
    "PerformanceMetrics",
    "compute_metrics",
    "CPCVResult",
    "CPCVEvaluator",
    "PermutationResult",
    "PermutationTest",
    "DSRResult",
    "DeflatedSharpeRatio",
    "MTRLResult",
    "MinTrackRecordLength",
    "ValidationReport",
    "BacktestValidator",
]
