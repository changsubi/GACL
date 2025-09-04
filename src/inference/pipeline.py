"""
Complete Wildlife Processing Pipeline

This module implements the complete two-stage pipeline for wildlife detection and classification.
"""

import torch
import cv2
import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import albumentations as A

from ..models.detection import WildlifeDetector
from ..models.gacl_model import GACLModel
from ..configs.config import config


class WildlifePipeline:
    """
    Complete Wildlife Detection and Classification Pipeline.
    
    This pipeline implements the two-stage approach described in the paper:
    Stage 1: Wildlife detection using YOLO/MegaDetector
    Stage 2: Species classification using GACL model
    
    The pipeline processes camera trap images to detect animals and classify
    them into Korean wildlife species categories.
    
    Args:
        detector_path (str): Path to detection model weights
        classifier_path (str): Path to trained GACL model weights
        confidence_threshold (float): Minimum confidence for classification
        device (str): Device to run inference on
    """
    
    def __init__(
        self,
        detector_path: str = None,
        classifier_path: str = None,
        confidence_threshold: float = None,
        device: str = None
    ):
        if detector_path is None:
            detector_path = config.detector_model_path
        if confidence_threshold is None:
            confidence_threshold = config.classification_confidence_threshold
        if device is None:
            device = str(config.device)
            
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Stage 1: Wildlife Detector
        self.detector = WildlifeDetector(
            model_path=detector_path,
            confidence_threshold=config.detection_confidence_threshold,
            device=device
        )
        
        # Stage 2: GACL Classifier
        self.gacl_model = GACLModel()
        if classifier_path and os.path.exists(classifier_path):
            self._load_classifier(classifier_path)
        else:
            self.logger.warning("No classifier model provided - detection only mode")
            self.gacl_model = None
        
        # Text processing components
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
        self._setup_text_prompts()
        
        # Image preprocessing
        self.transform = A.Compose([
            A.Resize(config.image_size, config.image_size)
        ])
        
        self.logger.info("Wildlife pipeline initialized")
    
    def _load_classifier(self, classifier_path: str):
        """Load trained GACL classifier."""
        try:
            checkpoint = torch.load(classifier_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.gacl_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.gacl_model.load_state_dict(checkpoint)
            
            self.gacl_model.to(self.device)
            self.gacl_model.eval()
            self.logger.info(f"Loaded GACL classifier from {classifier_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load classifier: {e}")
            self.gacl_model = None
    
    def _setup_text_prompts(self):
        """Setup text prompts for classification."""
        self.frame_prompts = {}
        self.object_prompts = {}
        
        for class_idx, class_name in enumerate(config.class_names):
            self.frame_prompts[class_idx] = config.frame_prompts[class_name]
            self.object_prompts[class_idx] = config.object_prompts[class_name]
    
    def process_single_image(
        self, 
        image_path: str,
        save_visualization: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path (str): Path to input image
            save_visualization (bool): Whether to save visualization
            output_dir (str, optional): Directory to save outputs
            
        Returns:
            Dict containing detection and classification results
        """
        try:
            # Stage 1: Wildlife Detection
            detections = self.detector.detect(image_path, return_crops=True)
            
            if not detections:
                return {
                    'image_path': image_path,
                    'wildlife_detected': False,
                    'message': 'No wildlife detected in image',
                    'detections': []
                }
            
            # Filter for animal detections
            animal_detections = self.detector.filter_detections(
                detections, animals_only=True
            )
            
            if not animal_detections:
                return {
                    'image_path': image_path,
                    'wildlife_detected': False,
                    'message': 'No animals detected in image',
                    'detections': detections  # Return all detections for reference
                }
            
            results = []
            
            # Stage 2: Species Classification for each detection
            if self.gacl_model is not None:
                for detection in animal_detections:
                    classification_result = self._classify_detection(detection)
                    
                    result = {
                        'bbox': detection['bbox'],
                        'detection_confidence': detection['confidence'],
                        'detection_class': detection['class_name'],
                        'species': classification_result['species'],
                        'species_confidence': classification_result['confidence'],
                        'species_probabilities': classification_result['probabilities']
                    }
                    results.append(result)
            else:
                # Detection only mode
                for detection in animal_detections:
                    result = {
                        'bbox': detection['bbox'],
                        'detection_confidence': detection['confidence'],
                        'detection_class': detection['class_name'],
                        'species': 'Unknown (No classifier loaded)',
                        'species_confidence': 0.0,
                        'species_probabilities': {}
                    }
                    results.append(result)
            
            # Create visualization if requested
            if save_visualization and output_dir:
                self._save_visualization(image_path, results, output_dir)
            
            return {
                'image_path': image_path,
                'wildlife_detected': True,
                'num_animals': len(results),
                'detections': results
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'wildlife_detected': False,
                'detections': []
            }
    
    def _classify_detection(self, detection: Dict) -> Dict:
        """
        Classify a single detection using GACL model.
        
        Args:
            detection (Dict): Detection dictionary with crop
            
        Returns:
            Dict: Classification results
        """
        if 'crop' not in detection:
            return {
                'species': 'Unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
        
        try:
            # Preprocess cropped image
            crop = detection['crop']
            transformed = self.transform(image=crop)
            image = transformed['image']
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Get classification scores for all classes
            class_scores = []
            
            with torch.no_grad():
                for class_idx in range(config.num_classes):
                    # Prepare text inputs
                    frame_prompt = self.frame_prompts[class_idx]
                    object_prompt = self.object_prompts[class_idx]
                    
                    frame_tokens = self.tokenizer(
                        frame_prompt, max_length=512, padding='max_length',
                        truncation=True, return_tensors='pt'
                    )
                    object_tokens = self.tokenizer(
                        object_prompt, max_length=100, padding='max_length',
                        truncation=True, return_tensors='pt'
                    )
                    
                    frame_ids = frame_tokens['input_ids'].to(self.device)
                    frame_mask = frame_tokens['attention_mask'].to(self.device)
                    object_ids = object_tokens['input_ids'].to(self.device)
                    object_mask = object_tokens['attention_mask'].to(self.device)
                    
                    # Get model predictions
                    outputs = self.gacl_model(
                        image=image_tensor,
                        frame_ids=frame_ids,
                        frame_mask=frame_mask,
                        object_ids=object_ids,
                        object_mask=object_mask,
                        training=False
                    )
                    
                    # Extract class score
                    probabilities = torch.softmax(outputs['logits'], dim=1)
                    class_scores.append(probabilities[0, class_idx].item())
            
            # Determine final prediction
            class_scores = torch.FloatTensor(class_scores)
            predicted_class = torch.argmax(class_scores).item()
            confidence = class_scores[predicted_class].item()
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                species_name = "Unidentified"
            else:
                species_name = config.class_names[predicted_class]
            
            # Create probability dictionary
            probabilities = {
                config.class_names[i]: class_scores[i].item()
                for i in range(config.num_classes)
            }
            
            return {
                'species': species_name,
                'confidence': confidence,
                'probabilities': probabilities
            }
            
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return {
                'species': 'Error',
                'confidence': 0.0,
                'probabilities': {}
            }
    
    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        save_visualizations: bool = True,
        save_results: bool = True
    ) -> Dict[str, Dict]:
        """
        Process multiple images through the pipeline.
        
        Args:
            image_paths (List[str]): List of image paths to process
            output_dir (str): Directory to save results
            save_visualizations (bool): Whether to save visualization images
            save_results (bool): Whether to save JSON results
            
        Returns:
            Dict: Results for all images
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        self.logger.info(f"Processing {len(image_paths)} images...")
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.process_single_image(
                    image_path=image_path,
                    save_visualization=save_visualizations,
                    output_dir=output_dir
                )
                results[image_path] = result
                
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results[image_path] = {
                    'error': str(e),
                    'wildlife_detected': False
                }
        
        # Save comprehensive results
        if save_results:
            self._save_batch_results(results, output_dir)
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(results)
        self.logger.info(f"Processing complete. Summary: {summary}")
        
        return results
    
    def _save_visualization(self, image_path: str, results: List[Dict], output_dir: str):
        """Save visualization of detection and classification results."""
        try:
            # Load original image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Draw results
            for i, result in enumerate(results):
                bbox = result['bbox']
                species = result['species']
                confidence = result['species_confidence']
                
                x1, y1, x2, y2 = bbox
                
                # Choose color based on species
                colors = {
                    'Wildboar': (255, 0, 0),    # Red
                    'Goral': (0, 255, 0),       # Green  
                    'Deers': (0, 0, 255),       # Blue
                    'Other': (255, 255, 0),     # Yellow
                    'Unidentified': (128, 128, 128)  # Gray
                }
                color = colors.get(species, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                # Draw label
                label = f"{species} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # Label background
                cv2.rectangle(
                    image,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                
                # Label text
                cv2.putText(
                    image, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2
                )
            
            # Save visualization
            filename = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(output_dir, f"{filename}_result.jpg")
            
            save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_image)
            
        except Exception as e:
            self.logger.error(f"Failed to save visualization for {image_path}: {e}")
    
    def _save_batch_results(self, results: Dict, output_dir: str):
        """Save batch processing results to JSON."""
        results_file = os.path.join(output_dir, 'classification_results.json')
        
        # Convert results to JSON-serializable format
        json_results = {}
        for image_path, result in results.items():
            json_results[image_path] = result
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _generate_summary_statistics(self, results: Dict) -> Dict:
        """Generate summary statistics from batch results."""
        total_images = len(results)
        wildlife_detected = sum(
            1 for r in results.values() 
            if r.get('wildlife_detected', False)
        )
        
        # Count species occurrences
        species_counts = {}
        total_detections = 0
        
        for result in results.values():
            if result.get('wildlife_detected', False):
                for detection in result.get('detections', []):
                    species = detection.get('species', 'Unknown')
                    species_counts[species] = species_counts.get(species, 0) + 1
                    total_detections += 1
        
        summary = {
            'total_images_processed': total_images,
            'images_with_wildlife': wildlife_detected,
            'wildlife_detection_rate': wildlife_detected / total_images if total_images > 0 else 0,
            'total_animal_detections': total_detections,
            'species_distribution': species_counts
        }
        
        # Save summary
        summary_file = os.path.join(
            os.path.dirname(list(results.keys())[0]) if results else '.',
            'processing_summary.json'
        )
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except:
            pass  # Don't fail if we can't save summary
        
        return summary
    
    def visualize_species_distribution(self, results: Dict, save_path: str):
        """Create and save species distribution visualization."""
        species_counts = {}
        
        for result in results.values():
            if result.get('wildlife_detected', False):
                for detection in result.get('detections', []):
                    species = detection.get('species', 'Unknown')
                    species_counts[species] = species_counts.get(species, 0) + 1
        
        if not species_counts:
            self.logger.info("No species data to visualize")
            return
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        species = list(species_counts.keys())
        counts = list(species_counts.values())
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        plt.bar(species, counts, color=colors[:len(species)])
        
        plt.title('Wildlife Species Distribution', fontsize=16)
        plt.xlabel('Species', fontsize=12)
        plt.ylabel('Number of Detections', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Species distribution plot saved to {save_path}")
