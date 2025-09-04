# GACL Wildlife Classification

A comprehensive implementation of **Graph Attention Contrastive Learning (GACL)** for Korean wildlife species classification using multi-modal learning. This project implements a two-stage detection and classification pipeline for camera trap image analysis.

## 🔬 Research Overview

This implementation is based on the GACL (Graph Attention Contrastive Learning) approach for fine-grained wildlife classification. The model combines:

- **Multi-Dilated Convolutional Networks** for multi-scale feature extraction
- **Graph Attention Transformer Encoders** for structural relationship learning
- **Parallel Contrastive Learning** for multi-modal alignment
- **Two-stage pipeline**: Detection (YOLO/MegaDetector) + Classification (GACL)

### Key Features

- ✅ **Multi-modal Learning**: Combines visual and textual representations
- ✅ **Graph-based Architecture**: Leverages spatial relationships in images
- ✅ **Contrastive Learning**: Four-way contrastive loss for robust learning
- ✅ **Korean Wildlife Focus**: Specialized for Korean species (Wildboar, Goral, Deers, Other)
- ✅ **Production Ready**: Complete training and inference pipelines
- ✅ **Comprehensive Evaluation**: Detailed metrics and visualization tools

## 📁 Project Structure

```
gacl_wildlife_classification/
├── src/                          # Source code
│   ├── configs/                  # Configuration management
│   │   └── config.py            # Main configuration file
│   ├── data/                    # Data loading and preprocessing
│   │   ├── dataset.py           # Dataset implementation
│   │   ├── augmentation.py      # Data augmentation
│   │   └── preprocessing.py     # Preprocessing utilities
│   ├── models/                  # Model architectures
│   │   ├── backbone.py          # Multi-dilated ConvNet & GAT Encoder
│   │   ├── encoders.py          # Vision and Text encoders
│   │   ├── losses.py            # Loss functions
│   │   ├── gacl_model.py        # Main GACL model
│   │   └── detection.py         # Wildlife detection module
│   ├── training/                # Training utilities
│   │   ├── trainer.py           # Main training loop
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── utils.py             # Training utilities
│   ├── inference/               # Inference pipelines
│   │   ├── pipeline.py          # Complete processing pipeline
│   │   └── predictor.py         # Simplified predictor interface
│   └── utils/                   # General utilities
│       └── dataset_utils.py     # Dataset creation and analysis
├── scripts/                     # Entry point scripts
│   ├── train.py                 # Training script
│   └── inference.py             # Inference script
├── configs/                     # Configuration files
├── docs/                        # Documentation
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd gacl_wildlife_classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended), install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Dataset Setup

#### Option A: Use Your Own Dataset

Organize your camera trap images in the following structure:

```
korean_wildlife_dataset/
├── train/
│   ├── Wildboar/
│   │   ├── image_001.jpg
│   │   └── ...
│   ├── Goral/
│   ├── Deers/
│   └── Other/
├── val/
│   └── [same structure as train]
└── test/
    └── [same structure as train]
```

**Required Image Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

**Security Note**: 🔒 The actual Korean wildlife dataset cannot be provided due to security restrictions. You must use your own camera trap images or create dummy data for testing.

#### Option B: Create Dummy Dataset for Testing

```bash
python scripts/train.py --create_dummy --dummy_samples 100
```

### 3. Training

#### Basic Training

```bash
python scripts/train.py --data_root ./korean_wildlife_dataset
```

#### Advanced Training Options

```bash
python scripts/train.py \
    --data_root ./korean_wildlife_dataset \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --model_save_path ./models \
    --results_path ./results \
    --device cuda
```

#### Resume Training from Checkpoint

```bash
python scripts/train.py \
    --data_root ./korean_wildlife_dataset \
    --resume_from ./models/checkpoint_epoch_50.pth
```

### 4. Inference

#### Full Pipeline (Detection + Classification)

```bash
python scripts/inference.py \
    --input ./test_images \
    --model_path ./models/best_model.pth \
    --detector_path yolov5s.pt \
    --output_dir ./inference_results \
    --save_visualizations
```

#### Classification Only (Pre-cropped Images)

```bash
python scripts/inference.py \
    --input ./cropped_animals \
    --model_path ./models/best_model.pth \
    --mode classifier_only \
    --output_dir ./classification_results
```

#### Single Image Inference

```bash
python scripts/inference.py \
    --input ./single_image.jpg \
    --model_path ./models/best_model.pth \
    --output_dir ./single_result
```

## 🏗️ Model Architecture

### GACL Model Components

1. **Multi-Dilated Convolutional Network**
   - Parallel convolutions with dilation rates [1, 2, 4, 5]
   - Captures features at different scales
   - Fine-grained to global information extraction

2. **Graph Attention Transformer Encoder**
   - Image patch extraction and graph construction
   - K-means clustering for semantic grouping
   - Multi-layer Graph Attention Networks with residual connections

3. **Vision Transformer Encoder**
   - Pre-trained ViT for local patch features
   - Global and local feature extraction

4. **BERT Text Encoder**
   - Processes frame-level and object-level text prompts
   - Semantic understanding of species descriptions

5. **Parallel Contrastive Learning Loss**
   - Four-way contrastive alignment:
     - Global Image ↔ Global Text (G-T)
     - Global Image ↔ Local Text (G-t)
     - Local Image ↔ Global Text (V-T)
     - Local Image ↔ Local Text (V-t)

### Text Prompts

The model uses carefully crafted text prompts for each species:

- **Wildboar**: "A large brown wild boar with coarse fur, stocky body, and prominent snout standing in forest habitat"
- **Goral**: "A small gray-brown goral with short curved horns, compact body, and agile stance on rocky terrain"
- **Deers**: "A graceful deer with slender legs, alert posture, and distinctive antlers or ears in woodland setting"
- **Other**: "Various small to medium mammals including raccoon dogs, badgers, and birds in natural habitat"

## 📊 Training Configuration

### Default Hyperparameters

```python
# Model Architecture
num_classes = 4
embedding_dim = 768
gat_hidden_dim = 256
gat_num_heads = 8
gat_num_layers = 3

# Training
batch_size = 16
num_epochs = 100
learning_rate = 1e-4
weight_decay = 1e-5
temperature = 0.07  # Contrastive learning temperature

# Data Augmentation
image_size = 224
mixup_alpha = 1.0
mixup_probability = 0.5
```

### Customizing Configuration

Modify `src/configs/config.py` to adjust model parameters, training settings, and data paths.

## 📈 Evaluation and Results

### Training Monitoring

The training process provides comprehensive monitoring:

- Real-time loss tracking (classification + contrastive)
- Validation accuracy and per-class metrics
- Confusion matrix generation
- Training curve visualization
- Early stopping and best model saving

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and macro-averaged metrics
- **Confusion Matrix**: Detailed classification analysis
- **Species Distribution**: Analysis of detection patterns

### Output Files

Training generates several output files:

```
models/
├── best_model.pth              # Best validation model
├── last_checkpoint.pth         # Latest checkpoint
└── checkpoint_epoch_X.pth      # Periodic checkpoints

results/
├── confusion_matrix.png        # Confusion matrix visualization
├── training_curves.png         # Loss and accuracy curves
├── evaluation_results.json     # Detailed evaluation metrics
└── training_history.json       # Complete training history
```

## 🔧 Advanced Usage

### Custom Dataset Integration

To use your own dataset:

1. **Organize your data** following the required structure
2. **Update text prompts** in `src/configs/config.py` for your species
3. **Adjust class names** and number of classes
4. **Modify augmentation** strategies if needed

### Model Customization

Key customization points:

- **Architecture modifications**: Edit `src/models/backbone.py`
- **Loss function changes**: Modify `src/models/losses.py`
- **Data augmentation**: Update `src/data/augmentation.py`
- **Training strategy**: Customize `src/training/trainer.py`

### Extending to New Species

To add new wildlife species:

1. Update `class_names` in config
2. Add corresponding text prompts
3. Retrain the model with new data
4. Adjust evaluation metrics

## 🧪 Testing and Validation

### Unit Tests

Run the test suite:

```bash
python -m pytest tests/
```

### Validation Scripts

Validate your dataset structure:

```python
from src.utils.dataset_utils import analyze_dataset_structure, validate_dataset_format

# Analyze dataset
analysis = analyze_dataset_structure('./korean_wildlife_dataset')
print(analysis)

# Validate format
is_valid, errors = validate_dataset_format('./korean_wildlife_dataset')
if not is_valid:
    print("Dataset issues:", errors)
```

## 📚 Research References

This implementation builds upon several key research areas:

### Referenced Models and Techniques

- **YOLO/YOLOv5**: For wildlife detection (Stage 1)
  - [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- **MegaDetector**: Camera trap animal detection
  - [Microsoft MegaDetector](https://github.com/microsoft/CameraTraps)
- **Vision Transformer (ViT)**: Image feature extraction
  - Dosovitskiy et al., "An Image is Worth 16x16 Words"
- **BERT**: Text encoding for prompts
  - Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers"
- **Graph Attention Networks**: Spatial relationship modeling
  - Veličković et al., "Graph Attention Networks"

### Core Techniques

- **Contrastive Learning**: Multi-modal representation alignment
- **Mixup Augmentation**: Data augmentation for robustness
- **Multi-scale Feature Extraction**: Dilated convolutions
- **Graph-based Learning**: Spatial relationship modeling

## 🚨 Important Notes

### Security and Data Privacy

⚠️ **Dataset Security**: The actual Korean wildlife dataset used in research cannot be provided due to security restrictions and wildlife protection policies. Users must:

- Use their own camera trap data
- Ensure proper permissions for wildlife imagery
- Follow local wildlife protection regulations
- Respect privacy and conservation guidelines

### Hardware Requirements

**Recommended Specifications**:
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4070 or better)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ for dataset and models
- **CPU**: Multi-core processor for data loading

**Minimum Specifications**:
- **GPU**: NVIDIA GPU with 4GB+ VRAM (GTX 1660 or better)
- **RAM**: 8GB+ system memory
- **Storage**: 20GB+ available space

### Performance Expectations

Training times (approximate):
- **Small dataset** (1K images/class): 2-4 hours on RTX 3070
- **Medium dataset** (5K images/class): 8-12 hours on RTX 3070
- **Large dataset** (15K images/class): 24-48 hours on RTX 3070

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows the style guidelines
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest mypy

# Run code formatting
black src/ scripts/

# Run linting
flake8 src/ scripts/

# Run type checking
mypy src/
```

## 📄 License

This project is licensed under an **Academic Use License** - see the LICENSE file for details.

**Important**: This software is intended for academic and research purposes only. Commercial use is strictly prohibited. For commercial licensing inquiries, please contact SPHERE AX.

## 🙏 Acknowledgments

- SPHERE AX AILab for the original GACL methodology(https://www.sphereax.com/)
- National Institute of Ecology for dataset guidance(https://www.nie.re.kr/)

---

**Note**: This implementation is for research and educational purposes. Please ensure compliance with local wildlife protection laws and ethical guidelines when working with wildlife imagery.



