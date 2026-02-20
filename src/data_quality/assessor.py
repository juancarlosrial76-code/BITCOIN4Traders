"""
Data Quality Assessment and Comparison Module
==============================================

Comprehensive data quality evaluation for financial data sources.

Features:
- Data completeness scoring
- Outlier detection and analysis
- Statistical distribution analysis
- Missing data pattern detection
- Data freshness assessment
- Cross-source comparison
- Quality scoring (0-100)
- Automated quality reports

Supports:
- Yahoo Finance
- Binance (CCXT)
- Alpha Vantage
- Quandl
- Local CSV files
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import skew, kurtosis, jarque_bera
import warnings

warnings.filterwarnings("ignore")


@dataclass
class DataQualityMetrics:
    """Data quality metrics container."""

    # Completeness
    completeness_score: float  # 0-100
    missing_values_pct: float
    missing_pattern: str  # 'random', 'clustered', 'systematic'

    # Consistency
    consistency_score: float  # 0-100
    duplicate_rows_pct: float
    timestamp_gaps: int
    gap_distribution: str  # 'uniform', 'irregular', 'none'

    # Accuracy
    accuracy_score: float  # 0-100
    outlier_pct: float
    outlier_severity: str  # 'low', 'medium', 'high'
    price_anomalies: int

    # Statistical Properties
    normality_score: float  # 0-100
    skewness: float
    kurtosis_val: float
    jarque_bera_pvalue: float

    # Freshness
    freshness_score: float  # 0-100
    data_age_hours: float
    update_frequency: str  # 'real-time', 'delayed', 'stale'

    # Overall
    overall_score: float  # 0-100
    quality_grade: str  # 'A', 'B', 'C', 'D', 'F'
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "completeness": {
                "score": round(self.completeness_score, 2),
                "missing_pct": round(self.missing_values_pct, 4),
                "pattern": self.missing_pattern,
            },
            "consistency": {
                "score": round(self.consistency_score, 2),
                "duplicates_pct": round(self.duplicate_rows_pct, 4),
                "gaps": self.timestamp_gaps,
                "gap_distribution": self.gap_distribution,
            },
            "accuracy": {
                "score": round(self.accuracy_score, 2),
                "outlier_pct": round(self.outlier_pct, 4),
                "severity": self.outlier_severity,
                "anomalies": self.price_anomalies,
            },
            "statistical": {
                "score": round(self.normality_score, 2),
                "skewness": round(self.skewness, 4),
                "kurtosis": round(self.kurtosis_val, 4),
                "jarque_bera_p": round(self.jarque_bera_pvalue, 4),
            },
            "freshness": {
                "score": round(self.freshness_score, 2),
                "age_hours": round(self.data_age_hours, 2),
                "frequency": self.update_frequency,
            },
            "overall": {
                "score": round(self.overall_score, 2),
                "grade": self.quality_grade,
            },
            "recommendations": self.recommendations,
        }


class DataQualityAssessor:
    """
    Assesses the quality of financial data.

    Provides comprehensive quality metrics and recommendations
    for improving data reliability in trading systems.
    """

    def __init__(self, df: pd.DataFrame, source_name: str = "unknown"):
        """
        Initialize quality assessor.

        Args:
            df: DataFrame with financial data
            source_name: Name of data source for reporting
        """
        self.df = df.copy()
        self.source_name = source_name
        self.metrics = None

        # Ensure timestamp column exists
        if "timestamp" not in self.df.columns and "date" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["date"])
        elif "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

    def assess(self) -> DataQualityMetrics:
        """Run complete quality assessment."""

        # Assess each dimension
        completeness = self._assess_completeness()
        consistency = self._assess_consistency()
        accuracy = self._assess_accuracy()
        statistical = self._assess_statistical_properties()
        freshness = self._assess_freshness()

        # Calculate overall score (weighted average)
        weights = {
            "completeness": 0.25,
            "consistency": 0.20,
            "accuracy": 0.25,
            "statistical": 0.15,
            "freshness": 0.15,
        }

        overall_score = (
            completeness["score"] * weights["completeness"]
            + consistency["score"] * weights["consistency"]
            + accuracy["score"] * weights["accuracy"]
            + statistical["score"] * weights["statistical"]
            + freshness["score"] * weights["freshness"]
        )

        # Determine grade
        grade = self._score_to_grade(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            completeness, consistency, accuracy, statistical, freshness
        )

        self.metrics = DataQualityMetrics(
            completeness_score=completeness["score"],
            missing_values_pct=completeness["missing_pct"],
            missing_pattern=completeness["pattern"],
            consistency_score=consistency["score"],
            duplicate_rows_pct=consistency["duplicates_pct"],
            timestamp_gaps=consistency["gaps"],
            gap_distribution=consistency["gap_distribution"],
            accuracy_score=accuracy["score"],
            outlier_pct=accuracy["outlier_pct"],
            outlier_severity=accuracy["severity"],
            price_anomalies=accuracy["anomalies"],
            normality_score=statistical["score"],
            skewness=statistical["skewness"],
            kurtosis_val=statistical["kurtosis"],
            jarque_bera_pvalue=statistical["jarque_bera_p"],
            freshness_score=freshness["score"],
            data_age_hours=freshness["age_hours"],
            update_frequency=freshness["frequency"],
            overall_score=overall_score,
            quality_grade=grade,
            recommendations=recommendations,
        )

        return self.metrics

    def _assess_completeness(self) -> Dict:
        """Assess data completeness."""
        # Calculate missing values percentage
        missing_counts = self.df.isnull().sum()
        total_cells = len(self.df) * len(self.df.columns)
        missing_total = missing_counts.sum()
        missing_pct = (missing_total / total_cells) * 100

        # Score: 100 if no missing, decreases with more missing
        score = max(0, 100 - (missing_pct * 5))  # 20% missing = 0 score

        # Detect missing pattern
        if missing_pct == 0:
            pattern = "none"
        elif missing_pct < 1:
            pattern = "random"
        elif missing_pct < 10:
            # Check if clustered
            missing_by_row = self.df.isnull().sum(axis=1)
            clustered = (missing_by_row > 0).astype(int).diff().abs().sum() < len(
                self.df
            ) * 0.1
            pattern = "clustered" if clustered else "random"
        else:
            pattern = "systematic"

        return {"score": score, "missing_pct": missing_pct, "pattern": pattern}

    def _assess_consistency(self) -> Dict:
        """Assess data consistency."""
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        duplicates_pct = (duplicates / len(self.df)) * 100

        # Check timestamp gaps
        gaps = 0
        gap_distribution = "none"

        if "timestamp" in self.df.columns:
            timestamps = pd.to_datetime(self.df["timestamp"]).sort_values()
            diffs = timestamps.diff().dropna()

            if len(diffs) > 0:
                expected_diff = (
                    diffs.mode()[0] if len(diffs.mode()) > 0 else diffs.median()
                )
                gaps = (diffs > expected_diff * 1.5).sum()

                # Determine gap distribution
                gap_std = diffs.std()
                gap_mean = diffs.mean()
                cv = gap_std / gap_mean if gap_mean > 0 else 0

                if cv < 0.1:
                    gap_distribution = "uniform"
                elif cv < 0.5:
                    gap_distribution = "irregular"
                else:
                    gap_distribution = "highly_irregular"

        # Score based on duplicates and gaps
        score = 100
        score -= min(30, duplicates_pct * 3)  # Up to 30 points for duplicates
        score -= min(40, gaps * 0.5)  # Up to 40 points for gaps
        score = max(0, score)

        return {
            "score": score,
            "duplicates_pct": duplicates_pct,
            "gaps": gaps,
            "gap_distribution": gap_distribution,
        }

    def _assess_accuracy(self) -> Dict:
        """Assess data accuracy using outlier detection."""
        outliers_pct = 0
        severity = "low"
        anomalies = 0

        # Check OHLCV data for outliers
        price_cols = ["open", "high", "low", "close"]
        available_cols = [col for col in price_cols if col in self.df.columns]

        if available_cols:
            # Use IQR method for outlier detection
            for col in available_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = (
                    (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                ).sum()
                outliers_pct = max(outliers_pct, (outliers / len(self.df)) * 100)

            # Check for price anomalies (e.g., close outside high/low)
            if all(col in self.df.columns for col in ["open", "high", "low", "close"]):
                anomalies += (self.df["close"] > self.df["high"]).sum()
                anomalies += (self.df["close"] < self.df["low"]).sum()
                anomalies += (self.df["open"] > self.df["high"]).sum()
                anomalies += (self.df["open"] < self.df["low"]).sum()

            # Determine severity
            if outliers_pct < 1:
                severity = "low"
            elif outliers_pct < 5:
                severity = "medium"
            else:
                severity = "high"

        # Score: 100 if <1% outliers, decreases with more outliers
        score = max(0, 100 - (outliers_pct * 10) - (anomalies * 2))

        return {
            "score": score,
            "outlier_pct": outliers_pct,
            "severity": severity,
            "anomalies": anomalies,
        }

    def _assess_statistical_properties(self) -> Dict:
        """Assess statistical properties of returns."""
        if "close" not in self.df.columns:
            return {"score": 50, "skewness": 0, "kurtosis": 3, "jarque_bera_p": 0.5}

        # Calculate returns
        returns = self.df["close"].pct_change().dropna()

        if len(returns) < 10:
            return {"score": 50, "skewness": 0, "kurtosis": 3, "jarque_bera_p": 0.5}

        # Calculate metrics
        skew_val = skew(returns)
        kurt_val = kurtosis(returns)

        # Jarque-Bera test for normality
        try:
            jb_stat, jb_pvalue = jarque_bera(returns)
        except:
            jb_pvalue = 0.5

        # Score based on normality (financial data shouldn't be perfectly normal)
        # But extreme skewness/kurtosis indicates data issues
        score = 100
        score -= min(30, abs(skew_val) * 10)  # Penalize extreme skewness
        score -= min(30, max(0, kurt_val - 3) * 5)  # Penalize excess kurtosis
        score = max(0, score)

        return {
            "score": score,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "jarque_bera_p": jb_pvalue,
        }

    def _assess_freshness(self) -> Dict:
        """Assess data freshness."""
        if "timestamp" not in self.df.columns:
            return {"score": 50, "age_hours": 0, "frequency": "unknown"}

        # Calculate data age
        latest_timestamp = pd.to_datetime(self.df["timestamp"]).max()
        now = datetime.now()
        age_hours = (now - latest_timestamp).total_seconds() / 3600

        # Determine update frequency
        timestamps = pd.to_datetime(self.df["timestamp"]).sort_values()
        if len(timestamps) > 1:
            diffs = timestamps.diff().dropna()
            median_diff = diffs.median()

            if median_diff < timedelta(minutes=1):
                frequency = "real-time"
            elif median_diff < timedelta(minutes=15):
                frequency = "high-frequency"
            elif median_diff < timedelta(hours=1):
                frequency = "hourly"
            elif median_diff < timedelta(days=1):
                frequency = "daily"
            else:
                frequency = "weekly"
        else:
            frequency = "unknown"

        # Score based on age
        if age_hours < 1:
            score = 100
        elif age_hours < 24:
            score = 90
        elif age_hours < 168:  # 1 week
            score = 70
        elif age_hours < 720:  # 1 month
            score = 50
        else:
            score = 20

        return {"score": score, "age_hours": age_hours, "frequency": frequency}

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _generate_recommendations(
        self, completeness, consistency, accuracy, statistical, freshness
    ) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []

        # Completeness recommendations
        if completeness["missing_pct"] > 5:
            recommendations.append(
                f"HIGH PRIORITY: {completeness['missing_pct']:.2f}% missing values detected. "
                "Consider data imputation or alternative data sources."
            )
        elif completeness["missing_pct"] > 0:
            recommendations.append(
                f"{completeness['missing_pct']:.2f}% missing values. "
                "Consider forward-fill for minor gaps."
            )

        # Consistency recommendations
        if consistency["duplicates_pct"] > 1:
            recommendations.append(
                f"Remove {consistency['duplicates_pct']:.2f}% duplicate rows to improve data integrity."
            )

        if consistency["gaps"] > 10:
            recommendations.append(
                f"{consistency['gaps']} timestamp gaps detected. "
                "Check data source reliability or consider interpolation."
            )

        # Accuracy recommendations
        if accuracy["outlier_pct"] > 5:
            recommendations.append(
                f"HIGH PRIORITY: {accuracy['outlier_pct']:.2f}% outliers detected ({accuracy['severity']} severity). "
                "Review data cleaning procedures."
            )
        elif accuracy["outlier_pct"] > 1:
            recommendations.append(
                f"{accuracy['outlier_pct']:.2f}% outliers present. "
                "Consider outlier detection in preprocessing."
            )

        if accuracy["anomalies"] > 0:
            recommendations.append(
                f"{accuracy['anomalies']} price anomalies detected (e.g., close outside high/low range). "
                "Verify data source accuracy."
            )

        # Freshness recommendations
        if freshness["score"] < 70:
            recommendations.append(
                f"Data is {freshness['age_hours']:.1f} hours old. "
                "Consider updating data source or switching to real-time feed."
            )

        if not recommendations:
            recommendations.append(
                "Data quality is excellent. No immediate action required."
            )

        return recommendations

    def print_report(self):
        """Print formatted quality report."""
        if self.metrics is None:
            self.assess()

        m = self.metrics

        print(f"\n{'=' * 70}")
        print(f"  DATA QUALITY REPORT: {self.source_name}")
        print(f"{'=' * 70}")

        print(
            f"\n  OVERALL QUALITY SCORE: {m.overall_score:.1f}/100 (Grade: {m.quality_grade})"
        )

        print(f"\n  1. COMPLETENESS ({m.completeness_score:.1f}/100)")
        print(f"     - Missing values: {m.missing_values_pct:.3f}%")
        print(f"     - Pattern: {m.missing_pattern}")

        print(f"\n  2. CONSISTENCY ({m.consistency_score:.1f}/100)")
        print(f"     - Duplicate rows: {m.duplicate_rows_pct:.3f}%")
        print(f"     - Timestamp gaps: {m.timestamp_gaps}")
        print(f"     - Gap distribution: {m.gap_distribution}")

        print(f"\n  3. ACCURACY ({m.accuracy_score:.1f}/100)")
        print(f"     - Outliers: {m.outlier_pct:.3f}% ({m.outlier_severity} severity)")
        print(f"     - Price anomalies: {m.price_anomalies}")

        print(f"\n  4. STATISTICAL PROPERTIES ({m.normality_score:.1f}/100)")
        print(f"     - Skewness: {m.skewness:.4f}")
        print(f"     - Kurtosis: {m.kurtosis_val:.4f}")
        print(f"     - Jarque-Bera p-value: {m.jarque_bera_pvalue:.4f}")

        print(f"\n  5. FRESHNESS ({m.freshness_score:.1f}/100)")
        print(f"     - Data age: {m.data_age_hours:.1f} hours")
        print(f"     - Update frequency: {m.update_frequency}")

        print(f"\n  RECOMMENDATIONS:")
        for i, rec in enumerate(m.recommendations, 1):
            print(f"     {i}. {rec}")

        print(f"\n{'=' * 70}\n")

    def get_quality_badge(self) -> str:
        """Get quality badge for documentation."""
        if self.metrics is None:
            self.assess()

        grade = self.metrics.quality_grade
        badges = {
            "A": "🟢 Excellent",
            "B": "🔵 Good",
            "C": "🟡 Acceptable",
            "D": "🟠 Poor",
            "F": "🔴 Unusable",
        }
        return badges.get(grade, "⚪ Unknown")


class DataSourceComparator:
    """
    Compare multiple data sources for the same asset/time period.

    Identifies discrepancies and recommends the best data source.
    """

    def __init__(self):
        """Initialize comparator."""
        self.sources: Dict[str, pd.DataFrame] = {}
        self.assessments: Dict[str, DataQualityMetrics] = {}

    def add_source(self, name: str, df: pd.DataFrame):
        """Add a data source for comparison."""
        self.sources[name] = df.copy()

        # Assess quality
        assessor = DataQualityAssessor(df, name)
        self.assessments[name] = assessor.assess()

    def compare(self) -> pd.DataFrame:
        """
        Compare all data sources.

        Returns:
            DataFrame with comparison metrics
        """
        if len(self.sources) < 2:
            raise ValueError("Need at least 2 sources to compare")

        # Create comparison dataframe
        comparison_data = []

        for name, metrics in self.assessments.items():
            comparison_data.append(
                {
                    "Source": name,
                    "Overall Score": metrics.overall_score,
                    "Grade": metrics.quality_grade,
                    "Completeness": metrics.completeness_score,
                    "Consistency": metrics.consistency_score,
                    "Accuracy": metrics.accuracy_score,
                    "Statistical": metrics.normality_score,
                    "Freshness": metrics.freshness_score,
                    "Missing %": metrics.missing_values_pct,
                    "Outliers %": metrics.outlier_pct,
                    "Data Age (h)": metrics.data_age_hours,
                    "Gaps": metrics.timestamp_gaps,
                }
            )

        return pd.DataFrame(comparison_data)

    def find_discrepancies(self, price_col: str = "close") -> Dict:
        """
        Find price discrepancies between sources.

        Returns:
            Dictionary with discrepancy statistics
        """
        if len(self.sources) < 2:
            return {}

        # Align data by timestamp
        aligned_data = {}
        for name, df in self.sources.items():
            if "timestamp" in df.columns or "date" in df.columns:
                ts_col = "timestamp" if "timestamp" in df.columns else "date"
                aligned_data[name] = df.set_index(pd.to_datetime(df[ts_col]))[price_col]

        if len(aligned_data) < 2:
            return {"error": "Could not align data sources"}

        # Create comparison dataframe
        comparison_df = pd.DataFrame(aligned_data)

        # Calculate discrepancies
        discrepancies = {}

        for i, source1 in enumerate(comparison_df.columns):
            for source2 in comparison_df.columns[i + 1 :]:
                pair_name = f"{source1}_vs_{source2}"

                # Calculate differences
                diff = comparison_df[source1] - comparison_df[source2]
                diff_pct = (diff / comparison_df[source2] * 100).abs()

                # Statistics
                discrepancies[pair_name] = {
                    "mean_diff": diff.mean(),
                    "mean_abs_diff": diff.abs().mean(),
                    "max_diff": diff.abs().max(),
                    "mean_pct_diff": diff_pct.mean(),
                    "max_pct_diff": diff_pct.max(),
                    "correlation": comparison_df[source1].corr(comparison_df[source2]),
                    "large_discrepancies": (diff_pct > 1.0).sum(),  # >1% difference
                }

        return discrepancies

    def recommend_best_source(self) -> Tuple[str, str]:
        """
        Recommend the best data source.

        Returns:
            Tuple of (best_source_name, reasoning)
        """
        if not self.assessments:
            raise ValueError("No sources added")

        # Score each source
        scores = {}
        for name, metrics in self.assessments.items():
            # Weighted score with penalty for poor freshness
            score = metrics.overall_score
            if metrics.freshness_score < 50:
                score *= 0.8  # 20% penalty for stale data
            scores[name] = score

        # Find best
        best_source = max(scores, key=scores.get)
        best_metrics = self.assessments[best_source]

        reasoning = (
            f"{best_source} scored {best_metrics.overall_score:.1f}/100 (Grade {best_metrics.quality_grade}) "
            f"with {best_metrics.completeness_score:.1f}% completeness, "
            f"{best_metrics.accuracy_score:.1f}% accuracy, and "
            f"{best_metrics.freshness_score:.1f}% freshness."
        )

        return best_source, reasoning

    def print_comparison_report(self):
        """Print comprehensive comparison report."""
        print(f"\n{'=' * 80}")
        print(f"  DATA SOURCE COMPARISON REPORT")
        print(f"{'=' * 80}")

        # Quality comparison
        print(f"\n  QUALITY SCORES:")
        comparison_df = self.compare()
        print(comparison_df.to_string(index=False))

        # Rankings
        print(f"\n  RANKINGS:")
        sorted_sources = sorted(
            self.assessments.items(), key=lambda x: x[1].overall_score, reverse=True
        )

        for rank, (name, metrics) in enumerate(sorted_sources, 1):
            badge = DataQualityAssessor(self.sources[name], name).get_quality_badge()
            print(f"     {rank}. {name}: {metrics.overall_score:.1f}/100 {badge}")

        # Discrepancies
        print(f"\n  PRICE DISCREPANCIES:")
        discrepancies = self.find_discrepancies()

        if "error" not in discrepancies:
            for pair, stats in discrepancies.items():
                print(f"\n     {pair}:")
                print(f"       - Mean difference: ${stats['mean_diff']:.4f}")
                print(f"       - Max difference: ${stats['max_diff']:.4f}")
                print(f"       - Mean % difference: {stats['mean_pct_diff']:.4f}%")
                print(
                    f"       - Large discrepancies (>{1.0}%): {stats['large_discrepancies']}"
                )
                print(f"       - Correlation: {stats['correlation']:.4f}")
        else:
            print(f"     Could not calculate discrepancies: {discrepancies['error']}")

        # Recommendation
        best_source, reasoning = self.recommend_best_source()
        print(f"\n  RECOMMENDATION:")
        print(f"     Best source: {best_source}")
        print(f"     Reasoning: {reasoning}")

        print(f"\n{'=' * 80}\n")


# Convenience functions
def assess_data_quality(
    df: pd.DataFrame, source_name: str = "unknown"
) -> DataQualityMetrics:
    """
    Quick function to assess data quality.

    Args:
        df: DataFrame to assess
        source_name: Name of data source

    Returns:
        DataQualityMetrics object
    """
    assessor = DataQualityAssessor(df, source_name)
    return assessor.assess()


def compare_data_sources(sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compare multiple data sources.

    Args:
        sources: Dictionary of {source_name: dataframe}

    Returns:
        Comparison DataFrame
    """
    comparator = DataSourceComparator()

    for name, df in sources.items():
        comparator.add_source(name, df)

    return comparator.compare()
