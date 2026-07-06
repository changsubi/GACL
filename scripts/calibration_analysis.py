#!/usr/bin/env python3
"""
Calibration and Threshold Analysis Script for GACL Wildlife Classification

This script implements the supplementary calibration analysis described in
Supporting Information S2 of the GACL paper. It examines the relationship
between predicted confidence and empirical accuracy, computes scalar
calibration metrics (Expected Calibration Error, Maximum Calibration Error,
Brier score, negative log-likelihood), renders a reliability diagram, and
produces a confidence-threshold trade-off analysis to support practical
operating-threshold selection for deployment.

The default confidence threshold of 0.8 (config.classification_confidence_threshold)
is highlighted throughout: predictions below this threshold are flagged for
manual review rather than discarded, reflecting the asymmetric costs of errors
in wildlife monitoring.

Three input modes are supported:

1. From saved evaluation results (recommended; no model/GPU required)::

       python scripts/calibration_analysis.py \
           --results_json ./results/evaluation_results.json \
           --output_dir ./calibration_results

2. From a labelled image directory, by running the trained classifier
   (requires torch, transformers, and the model checkpoint)::

       python scripts/calibration_analysis.py \
           --data_dir ./korean_wildlife_dataset/test \
           --model_path ./models/best_model.pth \
           --output_dir ./calibration_results

   The directory must contain one sub-folder per class (e.g. Wildboar/, Goral/,
   Deers/, Other/) matching config.class_names.

3. On synthetic demonstration data (no real data required), useful for
   verifying the tooling end-to-end::

       python scripts/calibration_analysis.py --synthetic \
           --output_dir ./calibration_results

Outputs written to ``--output_dir``:
    - reliability_diagram.png
    - threshold_analysis.png
    - calibration_metrics.json
    - threshold_analysis.csv
"""

import os
import sys
import argparse
import logging

import numpy as np

# Add both the src package and the repository root to the path. This mirrors
# scripts/inference.py (which adds src/) while also supporting the shipped
# layout where configs/config.py lives at the repository root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
sys.path.insert(0, os.path.join(_HERE, '..'))

from configs.config import config  # noqa: E402

# Import the calibration module directly from its file. This keeps the
# calibration tool dependency-light (numpy + matplotlib only) so that
# practitioners can compute calibration metrics on their own deployment data
# without pulling in the full training stack (torch, seaborn) that the
# ``training`` package __init__ would otherwise require.
import importlib.util  # noqa: E402

_calib_path = os.path.join(_HERE, '..', 'src', 'training', 'calibration.py')
_spec = importlib.util.spec_from_file_location('gacl_calibration', _calib_path)
_calibration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_calibration)
CalibrationAnalyzer = _calibration.CalibrationAnalyzer
generate_synthetic_predictions = _calibration.generate_synthetic_predictions


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GACL Calibration and Threshold Analysis (Supporting Information S2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input source (exactly one is required).
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--results_json', type=str, default=None,
        help='Path to evaluation_results.json produced by the trainer '
             '(contains probabilities, labels, class_names).'
    )
    source.add_argument(
        '--data_dir', type=str, default=None,
        help='Path to a labelled image directory with one sub-folder per class '
             '(runs the trained classifier to obtain probabilities).'
    )
    source.add_argument(
        '--synthetic', action='store_true',
        help='Use synthetic demonstration data instead of real predictions.'
    )

    # Required only for --data_dir mode.
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='Path to trained GACL model checkpoint (required with --data_dir).'
    )
    parser.add_argument(
        '--device', type=str, default=str(config.device),
        help='Device to use when running the classifier (cuda/cpu).'
    )

    # Analysis options.
    parser.add_argument(
        '--n_bins', type=int, default=15,
        help='Number of equal-width confidence bins for ECE/reliability.'
    )
    parser.add_argument(
        '--default_threshold', type=float,
        default=config.classification_confidence_threshold,
        help='Operating threshold to highlight in the analysis.'
    )
    parser.add_argument(
        '--thresholds', type=float, nargs='+', default=None,
        help='Explicit list of thresholds for the trade-off table '
             '(default: 0.50..0.95 step 0.05, plus 0.99).'
    )

    # Output and logging.
    parser.add_argument(
        '--output_dir', type=str, default='./calibration_results',
        help='Directory to save calibration figures and metrics.'
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level.'
    )

    return parser.parse_args()


def collect_predictions_from_directory(data_dir: str, model_path: str, device: str, logger):
    """
    Run the trained classifier over a labelled image directory and collect
    per-image probability vectors and ground-truth labels.

    The directory must contain one sub-folder per class in config.class_names.
    """
    import cv2  # local import: only needed for this mode
    from inference.predictor import WildlifePredictor

    class_to_idx = {name: i for i, name in enumerate(config.class_names)}
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    predictor = WildlifePredictor(model_path=model_path, device=device)

    probabilities = []
    labels = []
    for class_name in config.class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            logger.warning("Class sub-folder not found, skipping: %s", class_dir)
            continue

        image_files = sorted(
            f for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in image_exts
        )
        logger.info("Classifying %d images for class '%s'", len(image_files), class_name)

        for fname in image_files:
            path = os.path.join(class_dir, fname)
            image_bgr = cv2.imread(path)
            if image_bgr is None:
                logger.warning("Could not read image, skipping: %s", path)
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            result = predictor.predict_single(image_rgb, return_probabilities=True)
            prob_vector = [result['probabilities'][c] for c in config.class_names]
            probabilities.append(prob_vector)
            labels.append(class_to_idx[class_name])

    if not probabilities:
        raise RuntimeError(
            f"No labelled images found under {data_dir}. Expected sub-folders: "
            f"{config.class_names}"
        )

    return np.asarray(probabilities, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def build_analyzer(args, logger) -> CalibrationAnalyzer:
    """Construct a CalibrationAnalyzer from the selected input source."""
    if args.results_json is not None:
        logger.info("Loading predictions from %s", args.results_json)
        return CalibrationAnalyzer.from_evaluation_results(
            args.results_json, default_threshold=args.default_threshold
        )

    if args.data_dir is not None:
        if not args.model_path:
            raise ValueError("--model_path is required when using --data_dir.")
        logger.info("Running classifier over %s", args.data_dir)
        probs, labels = collect_predictions_from_directory(
            args.data_dir, args.model_path, args.device, logger
        )
        return CalibrationAnalyzer(
            probabilities=probs, labels=labels,
            class_names=config.class_names,
            default_threshold=args.default_threshold,
        )

    # Synthetic demonstration mode.
    logger.info("Generating synthetic demonstration predictions "
                "(no real data required).")
    probs, labels = generate_synthetic_predictions(
        n_samples=2000, num_classes=config.num_classes
    )
    return CalibrationAnalyzer(
        probabilities=probs, labels=labels,
        class_names=config.class_names,
        default_threshold=args.default_threshold,
    )


def main():
    """Main calibration analysis entry point."""
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting GACL Calibration and Threshold Analysis")
    logger.info("Arguments: %s", vars(args))

    try:
        analyzer = build_analyzer(args, logger)
    except Exception as e:
        logger.error("Failed to build calibration analyzer: %s", e)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Scalar metrics + human-readable summary.
    print(analyzer.summary(n_bins=args.n_bins))

    metrics_path = os.path.join(args.output_dir, 'calibration_metrics.json')
    analyzer.save_metrics(metrics_path, n_bins=args.n_bins)

    csv_path = os.path.join(args.output_dir, 'threshold_analysis.csv')
    analyzer.save_threshold_csv(csv_path)

    # Figures.
    reliability_path = os.path.join(args.output_dir, 'reliability_diagram.png')
    analyzer.plot_reliability_diagram(reliability_path, n_bins=args.n_bins)

    threshold_path = os.path.join(args.output_dir, 'threshold_analysis.png')
    analyzer.plot_threshold_analysis(threshold_path, thresholds=args.thresholds)

    logger.info("Calibration analysis completed. Results saved to: %s", args.output_dir)
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
