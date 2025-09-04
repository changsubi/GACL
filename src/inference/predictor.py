"""
Wildlife Predictor

Simplified inference interface for single predictions and batch processing.
"""

import torch
import numpy as np
from typing import Dict, List, Union, Optional
import logging

from ..models.gacl_model import GACLModel
from ..data.preprocessing import ImagePreprocessor, TextPreprocessor
from ..configs.config import config


class WildlifePredictor:
    """
    Simplified predictor interface for GACL model inference.
    
    This class provides a clean interface for making predictions
    without the full pipeline complexity.
    
    Args:
        model_path (str): Path to trained model checkpoint
        device (str, optional): Device to run inference on
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is None:
            device = str(config.device)
        
        self.device = device
        self.model_path = model_path
        
        # Initialize preprocessors
        self.image_processor = ImagePreprocessor()
        self.text_processor = TextPreprocessor()
        
        # Load model
        self.model = GACLModel()
        self._load_model()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Predictor initialized with model from {model_path}")
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        self.logger.info("Model loaded successfully")
    
    def predict_single(
        self,
        image: Union[str, np.ndarray],
        return_probabilities: bool = True,
        return_features: bool = False
    ) -> Dict:
        """
        Make prediction on a single image.
        
        Args:
            image: Image path or numpy array
            return_probabilities: Whether to return class probabilities
            return_features: Whether to return feature representations
            
        Returns:
            Dict containing prediction results
        """
        # Process image
        if isinstance(image, str):
            image_array = cv2.imread(image)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_array = image
        
        # Preprocess
        image_resized = self.image_processor.resize_image(image_array)
        image_tensor = self.image_processor.convert_to_tensor(image_resized)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Get text prompts for all classes
        frame_tokens_list = []
        object_tokens_list = []
        
        for class_name in config.class_names:
            frame_prompt = config.frame_prompts[class_name]
            object_prompt = config.object_prompts[class_name]
            
            # Tokenize
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
            
            frame_tokens = tokenizer(
                frame_prompt, max_length=512, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            object_tokens = tokenizer(
                object_prompt, max_length=100, padding='max_length', 
                truncation=True, return_tensors='pt'
            )
            
            frame_tokens_list.append(frame_tokens)
            object_tokens_list.append(object_tokens)
        
        # Make predictions for each class
        class_scores = []
        
        with torch.no_grad():
            for i in range(config.num_classes):
                frame_ids = frame_tokens_list[i]['input_ids'].to(self.device)
                frame_mask = frame_tokens_list[i]['attention_mask'].to(self.device)
                object_ids = object_tokens_list[i]['input_ids'].to(self.device)
                object_mask = object_tokens_list[i]['attention_mask'].to(self.device)
                
                outputs = self.model(
                    image=image_tensor,
                    frame_ids=frame_ids,
                    frame_mask=frame_mask,
                    object_ids=object_ids,
                    object_mask=object_mask,
                    training=False
                )
                
                # Get class probability
                probabilities = torch.softmax(outputs['logits'], dim=1)
                class_scores.append(probabilities[0, i].item())
        
        # Final prediction
        class_scores = torch.FloatTensor(class_scores)
        predicted_class_idx = torch.argmax(class_scores).item()
        confidence = class_scores[predicted_class_idx].item()
        predicted_class = config.class_names[predicted_class_idx]
        
        result = {
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence
        }
        
        if return_probabilities:
            result['probabilities'] = {
                config.class_names[i]: class_scores[i].item()
                for i in range(config.num_classes)
            }
        
        if return_features:
            # Get features for the predicted class
            best_frame_ids = frame_tokens_list[predicted_class_idx]['input_ids'].to(self.device)
            best_frame_mask = frame_tokens_list[predicted_class_idx]['attention_mask'].to(self.device)
            best_object_ids = object_tokens_list[predicted_class_idx]['input_ids'].to(self.device)
            best_object_mask = object_tokens_list[predicted_class_idx]['attention_mask'].to(self.device)
            
            features = self.model.get_feature_representations(
                image=image_tensor,
                frame_ids=best_frame_ids,
                frame_mask=best_frame_mask,
                object_ids=best_object_ids,
                object_mask=best_object_mask
            )
            result['features'] = features
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 8,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Make predictions on a batch of images.
        
        Args:
            images: List of image paths or numpy arrays
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            images = tqdm(images, desc="Making predictions")
        
        for image in images:
            try:
                result = self.predict_single(image)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Prediction failed for image: {e}")
                results.append({
                    'error': str(e),
                    'predicted_class': 'Error',
                    'confidence': 0.0
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        size_info = self.model.get_model_size()
        
        return {
            'model_path': self.model_path,
            'device': self.device,
            'num_classes': config.num_classes,
            'class_names': config.class_names,
            'model_parameters': size_info
        }
