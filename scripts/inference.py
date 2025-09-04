#!/usr/bin/env python3
"""
Inference Script for GACL Wildlife Classification

This script handles inference on new images using the trained GACL model.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from configs.config import config
from inference.pipeline import WildlifePipeline
from inference.predictor import WildlifePredictor


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GACL Wildlife Classification Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments
    parser.add_argument(
        '--input', type=str, required=True,
        help='Input image path or directory of images'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to trained GACL model checkpoint'
    )
    parser.add_argument(
        '--detector_path', type=str, default=config.detector_model_path,
        help='Path to YOLO detector model'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir', type=str, default='./inference_results',
        help='Directory to save inference results'
    )
    parser.add_argument(
        '--save_visualizations', action='store_true',
        help='Save visualization images with detection boxes'
    )
    parser.add_argument(
        '--save_crops', action='store_true',
        help='Save cropped animal regions'
    )
    
    # Processing options
    parser.add_argument(
        '--mode', type=str, choices=['pipeline', 'classifier_only'], 
        default='pipeline',
        help='Processing mode: full pipeline or classifier only'
    )
    parser.add_argument(
        '--confidence_threshold', type=float, 
        default=config.classification_confidence_threshold,
        help='Minimum confidence threshold for classification'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for processing (classifier_only mode)'
    )
    
    # Device and logging
    parser.add_argument(
        '--device', type=str, default=str(config.device),
        help='Device to use for inference (cuda/cpu)'
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def get_image_paths(input_path: str) -> list:
    """
    Get list of image paths from input (file or directory).
    
    Args:
        input_path (str): Path to image file or directory
        
    Returns:
        list: List of image file paths
    """
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return [str(input_path)]
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    elif input_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(input_path.glob(f'*{ext}'))
            image_paths.extend(input_path.glob(f'*{ext.upper()}'))
        
        return [str(p) for p in sorted(image_paths)]
    
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def run_pipeline_mode(args, image_paths: list, logger):
    """Run full detection + classification pipeline."""
    logger.info("Running full pipeline mode (detection + classification)")
    
    # Initialize pipeline
    pipeline = WildlifePipeline(
        detector_path=args.detector_path,
        classifier_path=args.model_path,
        confidence_threshold=args.confidence_threshold,
        device=args.device
    )
    
    # Process images
    if len(image_paths) == 1:
        # Single image
        logger.info(f"Processing single image: {image_paths[0]}")
        result = pipeline.process_single_image(
            image_path=image_paths[0],
            save_visualization=args.save_visualizations,
            output_dir=args.output_dir
        )
        
        # Print results
        print_single_result(result)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        result_file = os.path.join(args.output_dir, 'single_image_result.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
    else:
        # Batch processing
        logger.info(f"Processing {len(image_paths)} images")
        results = pipeline.process_batch(
            image_paths=image_paths,
            output_dir=args.output_dir,
            save_visualizations=args.save_visualizations,
            save_results=True
        )
        
        # Print summary
        print_batch_summary(results)
        
        # Create species distribution plot
        if results:
            plot_path = os.path.join(args.output_dir, 'species_distribution.png')
            pipeline.visualize_species_distribution(results, plot_path)


def run_classifier_only_mode(args, image_paths: list, logger):
    """Run classifier-only mode (assumes pre-cropped animal images)."""
    logger.info("Running classifier-only mode")
    
    # Initialize predictor
    predictor = WildlifePredictor(
        model_path=args.model_path,
        device=args.device
    )
    
    # Print model info
    model_info = predictor.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    results = []
    
    if len(image_paths) == 1:
        # Single image prediction
        logger.info(f"Classifying single image: {image_paths[0]}")
        result = predictor.predict_single(
            image=image_paths[0],
            return_probabilities=True
        )
        
        result['image_path'] = image_paths[0]
        results.append(result)
        
        # Print result
        print_classification_result(result)
        
    else:
        # Batch prediction
        logger.info(f"Classifying {len(image_paths)} images")
        batch_results = predictor.predict_batch(
            images=image_paths,
            batch_size=args.batch_size
        )
        
        for i, result in enumerate(batch_results):
            result['image_path'] = image_paths[i]
            results.append(result)
        
        # Print summary
        print_classification_summary(results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, 'classification_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)


def print_single_result(result: dict):
    """Print results for single image processing."""
    print("\n" + "="*60)
    print("WILDLIFE DETECTION AND CLASSIFICATION RESULTS")
    print("="*60)
    
    print(f"Image: {result['image_path']}")
    
    if result['wildlife_detected']:
        print(f"Wildlife detected: YES")
        print(f"Number of animals: {result['num_animals']}")
        
        for i, detection in enumerate(result['detections'], 1):
            print(f"\nAnimal {i}:")
            print(f"  Species: {detection['species']}")
            print(f"  Confidence: {detection['species_confidence']:.3f}")
            print(f"  Detection confidence: {detection['detection_confidence']:.3f}")
            print(f"  Bounding box: {detection['bbox']}")
            
            if 'species_probabilities' in detection:
                print("  Class probabilities:")
                for class_name, prob in detection['species_probabilities'].items():
                    print(f"    {class_name}: {prob:.3f}")
    else:
        print("Wildlife detected: NO")
        if 'message' in result:
            print(f"Message: {result['message']}")


def print_batch_summary(results: dict):
    """Print summary for batch processing."""
    total_images = len(results)
    wildlife_images = sum(1 for r in results.values() if r.get('wildlife_detected', False))
    
    species_counts = {}
    for result in results.values():
        if result.get('wildlife_detected', False):
            for detection in result.get('detections', []):
                species = detection.get('species', 'Unknown')
                species_counts[species] = species_counts.get(species, 0) + 1
    
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images processed: {total_images}")
    print(f"Images with wildlife: {wildlife_images}")
    print(f"Wildlife detection rate: {wildlife_images/total_images*100:.1f}%")
    
    if species_counts:
        print("\nSpecies distribution:")
        for species, count in sorted(species_counts.items()):
            print(f"  {species}: {count}")


def print_classification_result(result: dict):
    """Print classification result for single image."""
    print("\n" + "="*50)
    print("CLASSIFICATION RESULT")
    print("="*50)
    
    print(f"Image: {result['image_path']}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if 'probabilities' in result:
        print("\nClass probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.3f}")


def print_classification_summary(results: list):
    """Print summary for classification batch."""
    total_images = len(results)
    
    class_counts = {}
    for result in results:
        class_name = result.get('predicted_class', 'Error')
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\n" + "="*50)
    print("CLASSIFICATION SUMMARY")
    print("="*50)
    print(f"Total images classified: {total_images}")
    
    print("\nClass distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GACL Wildlife Classification Inference")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        return 1
    
    # Get image paths
    try:
        image_paths = get_image_paths(args.input)
        logger.info(f"Found {len(image_paths)} images to process")
    except Exception as e:
        logger.error(f"Error getting image paths: {e}")
        return 1
    
    if not image_paths:
        logger.error("No valid images found")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference based on mode
    try:
        if args.mode == 'pipeline':
            run_pipeline_mode(args, image_paths, logger)
        elif args.mode == 'classifier_only':
            run_classifier_only_mode(args, image_paths, logger)
        
        logger.info(f"Inference completed. Results saved to: {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
