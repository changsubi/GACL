"""
Calibration and Threshold Analysis Utilities

This module implements the supplementary calibration analysis described in
Supporting Information S2 of the GACL paper. It examines the relationship
between predicted confidence and empirical accuracy and supports practical
operating-threshold selection for deployment.

The default confidence threshold of ``0.8`` (see
``config.classification_confidence_threshold``) is used as a conservative
operating point for the camera-trap monitoring workflow: predictions with a
confidence below the threshold are flagged for manual review rather than being
discarded, so that uncertain cases can still be inspected by experts.

The utilities here are intentionally free of any heavy dependency (no ``torch``
required) so that practitioners can run calibration on their own deployment
data using only the model's output probabilities. Two entry points are
provided:

1. :class:`CalibrationAnalyzer` -- computes calibration metrics
   (Expected Calibration Error, Maximum Calibration Error, Brier score,
   negative log-likelihood), reliability curves, and threshold-coverage
   trade-off tables, and renders the corresponding figures.

2. :func:`temperature_scale` -- an optional post-hoc recalibration helper that
   fits a single temperature parameter on held-out logits.

Typical usage (from a saved ``evaluation_results.json`` file produced by
``WildlifeTrainer.final_evaluation``)::

    from training.calibration import CalibrationAnalyzer

    analyzer = CalibrationAnalyzer.from_evaluation_results(
        'results/evaluation_results.json'
    )
    analyzer.plot_reliability_diagram('results/reliability_diagram.png')
    analyzer.plot_threshold_analysis('results/threshold_analysis.png')
    print(analyzer.summary())
"""

import json
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level metric functions (operate on plain NumPy arrays)
# ---------------------------------------------------------------------------

def _as_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    """Validate and return an (N, C) probability matrix as float64."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(
            f"probabilities must be a 2D array of shape (N, num_classes); "
            f"got shape {probabilities.shape}"
        )
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        logger.warning(
            "Probability rows do not sum to 1 (min=%.4f, max=%.4f). "
            "Re-normalising each row.", row_sums.min(), row_sums.max()
        )
        probabilities = probabilities / np.clip(row_sums[:, None], 1e-12, None)
    return probabilities


def reliability_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, np.ndarray]:
    """
    Compute a top-label reliability curve using equal-width confidence bins.

    Args:
        confidences: Predicted confidence of the top class for each sample, in
            [0, 1], shape (N,).
        accuracies: Binary indicator (1.0 correct / 0.0 incorrect) for the top
            prediction of each sample, shape (N,).
        n_bins: Number of equal-width bins spanning [0, 1].

    Returns:
        Dict with per-bin arrays: ``bin_lowers``, ``bin_uppers``,
        ``bin_confidence`` (mean confidence in the bin), ``bin_accuracy``
        (empirical accuracy in the bin), ``bin_count`` (samples per bin), and
        ``bin_proportion`` (fraction of all samples in the bin). Empty bins are
        reported as NaN for confidence/accuracy and 0 for count/proportion.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    accuracies = np.asarray(accuracies, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    bin_confidence = np.full(n_bins, np.nan)
    bin_accuracy = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=np.int64)

    total = len(confidences)
    for i, (lo, hi) in enumerate(zip(bin_lowers, bin_uppers)):
        # Left-open, right-closed bins; the very first bin includes 0.0.
        if i == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum())
        bin_count[i] = count
        if count > 0:
            bin_confidence[i] = confidences[in_bin].mean()
            bin_accuracy[i] = accuracies[in_bin].mean()

    bin_proportion = bin_count / total if total > 0 else np.zeros(n_bins)

    return {
        'bin_lowers': bin_lowers,
        'bin_uppers': bin_uppers,
        'bin_confidence': bin_confidence,
        'bin_accuracy': bin_accuracy,
        'bin_count': bin_count,
        'bin_proportion': bin_proportion,
    }


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE), Guo et al. (2017).

    ECE is the sample-weighted average of the absolute gap between empirical
    accuracy and mean confidence across equal-width confidence bins:

        ECE = sum_m (|B_m| / N) * | acc(B_m) - conf(B_m) |
    """
    curve = reliability_curve(confidences, accuracies, n_bins=n_bins)
    gaps = np.abs(curve['bin_accuracy'] - curve['bin_confidence'])
    weights = curve['bin_proportion']
    valid = ~np.isnan(gaps)
    return float(np.sum(gaps[valid] * weights[valid]))


def maximum_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Maximum Calibration Error (MCE): the largest accuracy-confidence gap over
    all non-empty bins. Useful for risk-sensitive deployments where the
    worst-case miscalibration matters more than the average.
    """
    curve = reliability_curve(confidences, accuracies, n_bins=n_bins)
    gaps = np.abs(curve['bin_accuracy'] - curve['bin_confidence'])
    valid = ~np.isnan(gaps)
    return float(np.max(gaps[valid])) if np.any(valid) else 0.0


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """
    Multi-class Brier score: the mean squared error between the full predicted
    probability vector and the one-hot ground-truth vector.

        Brier = (1 / N) * sum_i sum_k (p_ik - y_ik)^2

    Ranges in [0, 2]; lower is better.
    """
    probabilities = _as_probability_matrix(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = probabilities.shape[1]
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def negative_log_likelihood(
    probabilities: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Mean negative log-likelihood (cross-entropy) of the true class."""
    probabilities = _as_probability_matrix(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    true_class_prob = probabilities[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_class_prob, eps, 1.0))))


def threshold_analysis(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """
    Trade-off between throughput and reliability as the acceptance threshold
    varies.

    For each threshold ``t`` a prediction is *auto-accepted* when its
    confidence is ``>= t``; otherwise it is flagged for manual review. Lower
    thresholds increase throughput (coverage) but may reduce the reliability of
    automatically accepted predictions, whereas higher thresholds improve the
    reliability of accepted predictions at the cost of more manual review.

    Args:
        confidences: Top-class confidence per sample, shape (N,).
        accuracies: Binary correctness of the top prediction, shape (N,).
        thresholds: Iterable of thresholds in [0, 1]. Defaults to
            ``0.50, 0.55, ..., 0.95, 0.99``.

    Returns:
        A list of dicts, one per threshold, with keys:
        ``threshold``, ``coverage`` (fraction auto-accepted),
        ``accepted_accuracy`` (accuracy among auto-accepted predictions),
        ``review_fraction`` (fraction flagged for manual review),
        ``n_accepted``, ``n_review``, and ``overall_accuracy_with_review``
        (accuracy assuming flagged cases are corrected by human review, i.e. an
        upper bound on end-to-end accuracy).
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    accuracies = np.asarray(accuracies, dtype=np.float64)
    total = len(confidences)

    if thresholds is None:
        thresholds = list(np.round(np.arange(0.50, 0.96, 0.05), 2)) + [0.99]

    rows: List[Dict[str, float]] = []
    for t in thresholds:
        accepted = confidences >= t
        n_accepted = int(accepted.sum())
        n_review = total - n_accepted
        coverage = n_accepted / total if total > 0 else 0.0
        accepted_accuracy = (
            float(accuracies[accepted].mean()) if n_accepted > 0 else float('nan')
        )
        # Optimistic end-to-end accuracy: auto-accepted predictions kept as-is,
        # flagged predictions assumed corrected by an expert reviewer.
        n_accepted_correct = float(accuracies[accepted].sum()) if n_accepted > 0 else 0.0
        overall_with_review = (n_accepted_correct + n_review) / total if total > 0 else 0.0
        rows.append({
            'threshold': float(t),
            'coverage': coverage,
            'accepted_accuracy': accepted_accuracy,
            'review_fraction': n_review / total if total > 0 else 0.0,
            'n_accepted': n_accepted,
            'n_review': n_review,
            'overall_accuracy_with_review': overall_with_review,
        })
    return rows


# ---------------------------------------------------------------------------
# High-level analyzer
# ---------------------------------------------------------------------------

class CalibrationAnalyzer:
    """
    Compute and visualise calibration metrics and threshold trade-offs for a
    set of model predictions.

    Args:
        probabilities: (N, num_classes) array of predicted class probabilities.
        labels: (N,) array of integer ground-truth class indices.
        class_names: Optional list of class names for reporting/plots.
        default_threshold: Operating threshold highlighted in the analysis
            (defaults to 0.8, matching
            ``config.classification_confidence_threshold``).
    """

    def __init__(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[List[str]] = None,
        default_threshold: float = 0.8,
    ):
        self.probabilities = _as_probability_matrix(probabilities)
        self.labels = np.asarray(labels, dtype=np.int64)
        if len(self.labels) != len(self.probabilities):
            raise ValueError(
                f"Number of labels ({len(self.labels)}) does not match number "
                f"of probability rows ({len(self.probabilities)})."
            )
        self.num_classes = self.probabilities.shape[1]
        self.class_names = class_names or [f'class_{i}' for i in range(self.num_classes)]
        self.default_threshold = float(default_threshold)

        # Top-label confidence and correctness derived once.
        self.predictions = self.probabilities.argmax(axis=1)
        self.confidences = self.probabilities.max(axis=1)
        self.accuracies = (self.predictions == self.labels).astype(np.float64)

    # ---- constructors ----------------------------------------------------

    @classmethod
    def from_evaluation_results(
        cls,
        results_path: str,
        default_threshold: float = 0.8,
    ) -> 'CalibrationAnalyzer':
        """
        Build an analyzer from an ``evaluation_results.json`` file as written by
        :meth:`WildlifeTrainer._save_evaluation_results`. The file is expected
        to contain ``probabilities`` (list of per-class probability vectors),
        ``labels`` (list of int), and optionally ``class_names``.
        """
        with open(results_path, 'r') as f:
            data = json.load(f)

        if 'probabilities' not in data or 'labels' not in data:
            raise KeyError(
                f"'{results_path}' must contain 'probabilities' and 'labels' "
                f"keys. Found keys: {list(data.keys())}"
            )

        return cls(
            probabilities=np.asarray(data['probabilities'], dtype=np.float64),
            labels=np.asarray(data['labels'], dtype=np.int64),
            class_names=data.get('class_names'),
            default_threshold=default_threshold,
        )

    # ---- scalar metrics --------------------------------------------------

    def compute_metrics(self, n_bins: int = 15) -> Dict[str, float]:
        """Return all scalar calibration metrics as a dictionary."""
        return {
            'n_samples': int(len(self.labels)),
            'n_bins': int(n_bins),
            'accuracy': float(self.accuracies.mean()),
            'mean_confidence': float(self.confidences.mean()),
            'ece': expected_calibration_error(self.confidences, self.accuracies, n_bins),
            'mce': maximum_calibration_error(self.confidences, self.accuracies, n_bins),
            'brier_score': brier_score(self.probabilities, self.labels),
            'nll': negative_log_likelihood(self.probabilities, self.labels),
            # Overconfidence = mean_confidence - accuracy (positive => overconfident).
            'overconfidence': float(self.confidences.mean() - self.accuracies.mean()),
        }

    def reliability_curve(self, n_bins: int = 15) -> Dict[str, np.ndarray]:
        """Per-bin reliability statistics (see module-level function)."""
        return reliability_curve(self.confidences, self.accuracies, n_bins=n_bins)

    def threshold_table(
        self, thresholds: Optional[Sequence[float]] = None
    ) -> List[Dict[str, float]]:
        """Threshold/coverage trade-off table (see module-level function)."""
        return threshold_analysis(self.confidences, self.accuracies, thresholds)

    # ---- reporting -------------------------------------------------------

    def summary(self, n_bins: int = 15) -> str:
        """Human-readable multi-line summary of the calibration analysis."""
        m = self.compute_metrics(n_bins=n_bins)
        lines = [
            "=" * 60,
            "CALIBRATION AND THRESHOLD ANALYSIS (Supporting Information S2)",
            "=" * 60,
            f"Samples                : {m['n_samples']}",
            f"Classes                : {self.num_classes} {self.class_names}",
            f"Overall accuracy       : {m['accuracy']*100:.2f}%",
            f"Mean confidence        : {m['mean_confidence']*100:.2f}%",
            f"Overconfidence gap     : {m['overconfidence']*100:+.2f} pp",
            "-" * 60,
            f"Expected Calibration Error (ECE, {n_bins} bins): {m['ece']:.4f}",
            f"Maximum  Calibration Error (MCE, {n_bins} bins): {m['mce']:.4f}",
            f"Brier score (multi-class)               : {m['brier_score']:.4f}",
            f"Negative log-likelihood                 : {m['nll']:.4f}",
            "-" * 60,
            f"Threshold trade-off (default threshold = {self.default_threshold}):",
            f"{'thresh':>7} {'coverage':>10} {'acc(accepted)':>14} {'review':>9}",
        ]
        default_thresholds = sorted(set(
            [0.5, 0.6, 0.7, self.default_threshold, 0.9, 0.95]
        ))
        for row in self.threshold_table(default_thresholds):
            marker = "  <- default" if abs(row['threshold'] - self.default_threshold) < 1e-9 else ""
            acc = row['accepted_accuracy']
            acc_str = f"{acc*100:12.2f}%" if not np.isnan(acc) else f"{'n/a':>13}"
            lines.append(
                f"{row['threshold']:>7.2f} {row['coverage']*100:9.2f}% "
                f"{acc_str} {row['review_fraction']*100:8.2f}%{marker}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def save_metrics(self, path: str, n_bins: int = 15) -> None:
        """Save scalar metrics and the full threshold table to JSON."""
        payload = {
            'metrics': self.compute_metrics(n_bins=n_bins),
            'default_threshold': self.default_threshold,
            'threshold_analysis': self.threshold_table(),
            'class_names': self.class_names,
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        logger.info("Calibration metrics saved to %s", path)

    def save_threshold_csv(self, path: str) -> None:
        """Save the threshold trade-off table as CSV (no pandas dependency)."""
        rows = self.threshold_table()
        header = [
            'threshold', 'coverage', 'accepted_accuracy', 'review_fraction',
            'n_accepted', 'n_review', 'overall_accuracy_with_review',
        ]
        with open(path, 'w') as f:
            f.write(','.join(header) + '\n')
            for row in rows:
                f.write(','.join(f"{row[k]:.6f}" for k in header) + '\n')
        logger.info("Threshold table saved to %s", path)

    # ---- plotting --------------------------------------------------------

    def plot_reliability_diagram(
        self,
        save_path: Optional[str] = None,
        n_bins: int = 15,
        title: str = 'Reliability Diagram - GACL Wildlife Model',
    ):
        """
        Render a reliability diagram (accuracy vs. confidence) with a confidence
        histogram underneath and ECE/Brier annotations.

        Returns the Matplotlib figure. If ``save_path`` is given the figure is
        also written to disk at 300 DPI.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        curve = self.reliability_curve(n_bins=n_bins)
        metrics = self.compute_metrics(n_bins=n_bins)
        centers = (curve['bin_lowers'] + curve['bin_uppers']) / 2.0
        width = 1.0 / n_bins

        fig = plt.figure(figsize=(7, 8))
        gs = GridSpec(3, 1, height_ratios=[3, 1, 0.05], hspace=0.28)
        ax_rel = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])

        # --- Reliability panel ---
        ax_rel.plot([0, 1], [0, 1], linestyle='--', color='gray',
                    label='Perfect calibration')

        acc = curve['bin_accuracy']
        conf = curve['bin_confidence']
        valid = ~np.isnan(acc)

        # Bars: empirical accuracy per bin.
        ax_rel.bar(
            centers[valid], acc[valid], width=width, align='center',
            edgecolor='black', color='#4C72B0', alpha=0.85, label='Accuracy',
        )
        # Gap bars showing the accuracy-confidence discrepancy.
        gap = np.where(valid, conf - acc, 0.0)
        ax_rel.bar(
            centers[valid], gap[valid], width=width, align='center',
            bottom=acc[valid], edgecolor='#C44E52', color='#C44E52',
            alpha=0.35, hatch='//', label='Gap (conf - acc)',
        )
        ax_rel.axvline(
            self.default_threshold, color='green', linestyle=':', linewidth=2,
            label=f'Default threshold = {self.default_threshold}',
        )
        ax_rel.set_xlim(0, 1)
        ax_rel.set_ylim(0, 1)
        ax_rel.set_ylabel('Accuracy')
        ax_rel.set_title(title)
        ax_rel.legend(loc='upper left', fontsize=9)
        ax_rel.text(
            0.98, 0.05,
            f"ECE = {metrics['ece']:.4f}\nMCE = {metrics['mce']:.4f}\n"
            f"Brier = {metrics['brier_score']:.4f}",
            transform=ax_rel.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        )

        # --- Confidence histogram panel ---
        ax_hist.bar(
            centers, curve['bin_proportion'], width=width, align='center',
            edgecolor='black', color='#55A868', alpha=0.85,
        )
        ax_hist.axvline(
            self.confidences.mean(), color='black', linestyle='-', linewidth=1.5,
            label=f'Mean conf = {self.confidences.mean():.3f}',
        )
        ax_hist.axvline(
            self.accuracies.mean(), color='red', linestyle='--', linewidth=1.5,
            label=f'Accuracy = {self.accuracies.mean():.3f}',
        )
        ax_hist.axvline(self.default_threshold, color='green', linestyle=':', linewidth=2)
        ax_hist.set_xlim(0, 1)
        ax_hist.set_xlabel('Confidence')
        ax_hist.set_ylabel('Fraction')
        ax_hist.legend(loc='upper left', fontsize=9)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Reliability diagram saved to %s", save_path)
        return fig

    def plot_threshold_analysis(
        self,
        save_path: Optional[str] = None,
        thresholds: Optional[Sequence[float]] = None,
        title: str = 'Confidence Threshold Trade-off - GACL Wildlife Model',
    ):
        """
        Plot coverage (throughput) and accepted-prediction accuracy against the
        confidence threshold, with the default operating point highlighted.

        Returns the Matplotlib figure. If ``save_path`` is given the figure is
        also written to disk at 300 DPI.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if thresholds is None:
            thresholds = list(np.round(np.arange(0.05, 1.00, 0.05), 2))
        rows = self.threshold_table(thresholds)
        t = np.array([r['threshold'] for r in rows])
        coverage = np.array([r['coverage'] for r in rows])
        acc = np.array([r['accepted_accuracy'] for r in rows])

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color_cov = '#4C72B0'
        ax1.set_xlabel('Confidence threshold')
        ax1.set_ylabel('Coverage (fraction auto-accepted)', color=color_cov)
        line_cov = ax1.plot(t, coverage, '-o', color=color_cov, label='Coverage')
        ax1.tick_params(axis='y', labelcolor=color_cov)
        ax1.set_ylim(0, 1.02)

        ax2 = ax1.twinx()
        color_acc = '#C44E52'
        ax2.set_ylabel('Accuracy of auto-accepted predictions', color=color_acc)
        line_acc = ax2.plot(t, acc, '-s', color=color_acc, label='Accepted accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)
        ax2.set_ylim(min(0.9, np.nanmin(acc)) if np.any(~np.isnan(acc)) else 0.0, 1.005)

        ax1.axvline(
            self.default_threshold, color='green', linestyle=':', linewidth=2,
            label=f'Default threshold = {self.default_threshold}',
        )

        # Combined legend.
        lines = line_cov + line_acc
        labels = [ln.get_label() for ln in lines]
        labels.append(f'Default threshold = {self.default_threshold}')
        from matplotlib.lines import Line2D
        lines.append(Line2D([0], [0], color='green', linestyle=':', linewidth=2))
        ax1.legend(lines, labels, loc='lower center', fontsize=9)
        ax1.set_title(title)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Threshold analysis figure saved to %s", save_path)
        return fig


# ---------------------------------------------------------------------------
# Optional post-hoc recalibration
# ---------------------------------------------------------------------------

def temperature_scale(
    logits: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 200,
) -> Tuple[float, np.ndarray]:
    """
    Fit a single temperature parameter ``T`` that rescales logits to minimise
    negative log-likelihood (Guo et al., 2017, "On Calibration of Modern Neural
    Networks"). This is an optional post-hoc recalibration step; it does not
    change the model's predictions (argmax is invariant to a positive
    temperature) but improves the reliability of the confidence estimates.

    Args:
        logits: (N, num_classes) array of pre-softmax scores from a held-out
            calibration split.
        labels: (N,) integer ground-truth class indices.
        lr: Learning-rate for the gradient-descent search over ``log T``.
        max_iter: Maximum number of optimisation steps.

    Returns:
        Tuple ``(temperature, calibrated_probabilities)`` where
        ``calibrated_probabilities`` is the (N, num_classes) softmax of
        ``logits / temperature``.

    Note:
        If only probabilities (not logits) are available, approximate logits can
        be recovered as ``log(clip(probabilities, eps, 1))``; the resulting
        temperature is an approximation because the additive softmax constant is
        lost.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)

    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # Optimise over log_T so temperature stays strictly positive.
    log_t = 0.0
    for _ in range(max_iter):
        t = np.exp(log_t)
        probs = _softmax(logits / t)
        true_probs = probs[np.arange(n), labels]
        # d(NLL)/d(log_T): closed form via chain rule through the softmax.
        # grad = mean over samples of ( sum_k p_k * s_k - s_y ), where
        # s_k = logit_k / T is the scaled logit, times d(scaled)/d(log_T) = -s_k.
        scaled = logits / t
        expected_scaled = np.sum(probs * scaled, axis=1)
        true_scaled = scaled[np.arange(n), labels]
        grad = np.mean(-(expected_scaled - true_scaled))  # d NLL / d log_T
        log_t -= lr * grad
        if abs(grad) < 1e-7:
            break

    temperature = float(np.exp(log_t))
    calibrated = _softmax(logits / temperature)
    return temperature, calibrated


# ---------------------------------------------------------------------------
# Synthetic data for testing / demonstration (no real data required)
# ---------------------------------------------------------------------------

def generate_synthetic_predictions(
    n_samples: int = 2000,
    num_classes: int = 4,
    overconfidence: float = 1.4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (probabilities, labels) that mimic a moderately
    overconfident classifier, so the calibration tools can be exercised and
    demonstrated without access to the (non-public) Korean Wildlife Dataset.

    Args:
        n_samples: Number of synthetic samples.
        num_classes: Number of classes.
        overconfidence: Sharpening exponent applied to the softmax; values > 1
            make the classifier overconfident (a realistic failure mode), 1.0
            is neutral, and < 1 makes it under-confident.
        seed: RNG seed for reproducibility.

    Returns:
        ``(probabilities, labels)`` with shapes (n_samples, num_classes) and
        (n_samples,).
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=n_samples)

    # Latent scores with the true class favoured by a random margin.
    logits = rng.normal(0.0, 1.0, size=(n_samples, num_classes))
    margin = rng.normal(2.0, 1.0, size=n_samples)
    logits[np.arange(n_samples), labels] += margin

    # Sharpen to induce overconfidence, then softmax.
    logits *= overconfidence
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    return probabilities, labels
