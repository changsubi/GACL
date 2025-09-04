"""
Wildlife Detection Module

This module implements the first stage of the two-stage pipeline:
wildlife detection using YOLO/MegaDetector approach.
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from ultralytics import YOLO

from ..configs.config import config


class WildlifeDetector:
    """
    Wildlife Detection using YOLO/MegaDetector approach.
    
    This class implements the first stage of the pipeline where we detect
    potential wildlife/animals in camera trap images before classification.
    
    The detector uses a pre-trained YOLO model or MegaDetector weights
    to identify regions of interest containing animals.
    
    Args:
        model_path (str): Path to YOLO model weights
        confidence_threshold (float): Minimum confidence for detections
        device (str): Device to run inference on
    """
    
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = None,
        device: str = None
    ):
        if model_path is None:
            model_path = config.detector_model_path
        if confidence_threshold is None:
            confidence_threshold = config.detection_confidence_threshold
        if device is None:
            device = str(config.device)
            
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Loaded detection model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load detection model: {e}")
            raise
        
        # Define animal class names (COCO dataset classes that are animals)
        self.animal_classes = {
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
            'bear', 'zebra', 'giraffe', 'deer', 'wildlife', 'animal'
        }
        
        # Additional classes that might indicate wildlife presence
        self.potential_wildlife_classes = {
            'person',  # Might be in wildlife images
            'vehicle'  # Camera traps might catch vehicles
        }
    
    def detect(
        self, 
        image_path: str,
        return_crops: bool = False,
        min_area: int = 1000
    ) -> List[Dict]:
        """
        Detect wildlife/animals in a single image.
        
        Args:
            image_path (str): Path to input image
            return_crops (bool): Whether to return cropped regions
            min_area (int): Minimum bounding box area to consider
            
        Returns:
            List of detection dictionaries containing bbox, confidence, class, etc.
        """
        try:
            # Run YOLO detection
            results = self.model(image_path, verbose=False)
            
            detections = []
            original_image = None
            
            if return_crops:
                original_image = cv2.imread(image_path)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        conf = box.conf[0].item()
                        cls_id = int(box.cls[0].item())
                        class_name = self.model.names[cls_id]
                        
                        # Filter based on confidence and class
                        if conf >= self.confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Check minimum area
                            area = (x2 - x1) * (y2 - y1)
                            if area < min_area:
                                continue
                            
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': conf,
                                'class_id': cls_id,
                                'class_name': class_name,
                                'area': area,
                                'is_animal': self._is_animal_class(class_name)
                            }
                            
                            # Add cropped image if requested
                            if return_crops and original_image is not None:
                                crop = self._crop_detection(original_image, detection['bbox'])
                                detection['crop'] = crop
                            
                            detections.append(detection)
            
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed for {image_path}: {e}")
            return []
    
    def detect_batch(
        self, 
        image_paths: List[str],
        return_crops: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[Dict]]:
        """
        Detect wildlife in multiple images.
        
        Args:
            image_paths (List[str]): List of image paths
            return_crops (bool): Whether to return cropped regions
            progress_callback (callable, optional): Progress callback function
            
        Returns:
            Dict mapping image paths to their detection results
        """
        results = {}
        
        for i, image_path in enumerate(image_paths):
            try:
                detections = self.detect(image_path, return_crops=return_crops)
                results[image_path] = detections
                
                if progress_callback:
                    progress_callback(i + 1, len(image_paths))
                    
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results[image_path] = []
        
        return results
    
    def _is_animal_class(self, class_name: str) -> bool:
        """
        Check if a class name corresponds to an animal.
        
        Args:
            class_name (str): YOLO class name
            
        Returns:
            bool: True if class is considered an animal
        """
        class_name_lower = class_name.lower()
        return (
            class_name_lower in self.animal_classes or
            any(animal in class_name_lower for animal in self.animal_classes)
        )
    
    def _crop_detection(
        self, 
        image: np.ndarray, 
        bbox: List[int],
        padding: float = 0.1
    ) -> np.ndarray:
        """
        Crop detection region with optional padding.
        
        Args:
            image (np.ndarray): Original image
            bbox (List[int]): Bounding box [x1, y1, x2, y2]
            padding (float): Padding ratio around bounding box
            
        Returns:
            np.ndarray: Cropped image region
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Calculate padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)
        
        # Apply padding with bounds checking
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)
        
        return image[y1_pad:y2_pad, x1_pad:x2_pad]
    
    def filter_detections(
        self,
        detections: List[Dict],
        animals_only: bool = True,
        min_confidence: float = None,
        max_detections: int = None
    ) -> List[Dict]:
        """
        Filter detections based on various criteria.
        
        Args:
            detections (List[Dict]): List of detection dictionaries
            animals_only (bool): Keep only animal detections
            min_confidence (float, optional): Minimum confidence threshold
            max_detections (int, optional): Maximum number of detections to keep
            
        Returns:
            List[Dict]: Filtered detections
        """
        filtered = detections.copy()
        
        # Filter by animal class
        if animals_only:
            filtered = [d for d in filtered if d.get('is_animal', False)]
        
        # Filter by confidence
        if min_confidence is not None:
            filtered = [d for d in filtered if d['confidence'] >= min_confidence]
        
        # Limit number of detections
        if max_detections is not None:
            filtered = filtered[:max_detections]
        
        return filtered
    
    def visualize_detections(
        self,
        image_path: str,
        detections: List[Dict],
        save_path: Optional[str] = None,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image_path (str): Path to original image
            detections (List[Dict]): Detection results
            save_path (str, optional): Path to save visualization
            show_confidence (bool): Whether to show confidence scores
            
        Returns:
            np.ndarray: Image with detection visualizations
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            is_animal = detection.get('is_animal', False)
            
            x1, y1, x2, y2 = bbox
            
            # Choose color based on whether it's an animal
            color = (0, 255, 0) if is_animal else (255, 255, 0)  # Green for animals, yellow for others
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}"
            if show_confidence:
                label += f" {confidence:.2f}"
            
            # Calculate label size and position
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y = max(y1, label_size[1] + 10)
            
            # Draw label background
            cv2.rectangle(
                image,
                (x1, label_y - label_size[1] - 10),
                (x1 + label_size[0], label_y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image, label,
                (x1, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        # Save if path provided
        if save_path:
            save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_image)
            self.logger.info(f"Saved detection visualization to {save_path}")
        
        return image
    
    def get_detection_statistics(self, detections: List[Dict]) -> Dict:
        """
        Get statistics about detections.
        
        Args:
            detections (List[Dict]): Detection results
            
        Returns:
            Dict: Detection statistics
        """
        if not detections:
            return {
                'total_detections': 0,
                'animal_detections': 0,
                'avg_confidence': 0.0,
                'classes': {}
            }
        
        animal_count = sum(1 for d in detections if d.get('is_animal', False))
        avg_confidence = np.mean([d['confidence'] for d in detections])
        
        # Count classes
        class_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_detections': len(detections),
            'animal_detections': animal_count,
            'avg_confidence': float(avg_confidence),
            'classes': class_counts,
            'confidence_range': {
                'min': float(min(d['confidence'] for d in detections)),
                'max': float(max(d['confidence'] for d in detections))
            }
        }
