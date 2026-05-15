# Flower Image Classifier Project

## Project Overview

This project implements a deep learning-based flower image classifier that can identify 102 different species of flowers. The classifier uses transfer learning with MobileNet V3 as the base model, combined with custom fully connected layers for classification.

**Author:** Layan Buirat  
**Course:** Udacity Academy  
**Development Period:** April 7–15, 2026 (Notebook) & May 7–15, 2026 (Predict Script)

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Usage](#usage)
- [Results](#results)
- [Files Description](#files-description)
- [Command Line Arguments](#command-line-arguments)

## Project Structure

```
flower-classifier/
├── layan-classification-image.ipynb   # Main Jupyter notebook for training
├── predict.py                          # Command line prediction script
├── label_map.json                      # Label mapping (102 flower species)
├── requirements.txt                    # Python dependencies
├── saved_models/                       # Saved model files (HDF5 format)
└── test_images/                        # Test images folder
    ├── cautleya_spicata.jpg
    ├── hard-leaved_pocket_orchid.jpg
    ├── orange_dahlia.jpg
    └── wild_pansy.jpg
```

## Dataset

The project uses the **Oxford Flowers 102 dataset** from TensorFlow Datasets. This dataset contains:

- **102 flower categories** (102 classes)
- **Training set:** 1,020 images
- **Validation set:** 1,020 images  
- **Test set:** 6,149 images

The dataset includes common and rare flower species from the UK, with each image being a photo of a flower belonging to one of the 102 categories.

## Requirements

### Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```
tensorflow>=2.19.0
tensorflow-hub>=0.16.1
tensorflow-datasets>=4.9.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=9.5.0
```

### Hardware Requirements

- **GPU recommended** for training (tested on Tesla P100-PCIE-16GB)
- Minimum 8GB RAM for data loading
- ~2GB storage space for model and dataset

## Model Architecture

The classifier uses transfer learning with the following architecture:

1. **Base Model:** MobileNet V3 Large (0.75x) from TensorFlow Hub
   - Pre-trained on ImageNet
   - Feature vector output dimension: 1280
   - Set as non-trainable

2. **Custom Classification Head:**
   - Dense layer: 1000 units with ReLU activation
   - Dropout layer: 0.5 dropout rate for regularization
   - Output layer: 102 units with softmax activation

### Model Summary

```
Total params: 1,383,102 (5.28 MB)
Trainable params: 1,383,102
Non-trainable params: 0
```

## Training

### Training Configuration

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Batch Size:** 32
- **Image Size:** 224×224 pixels
- **Epochs:** 10-50 (optimal at 10 epochs)

### Data Preprocessing

Images are preprocessed as follows:
1. Resized to 224×224 pixels
2. Pixel values normalized to [0, 1] range
3. Batched with prefetching for performance

### Training Results

After 10 epochs:
- **Validation Accuracy:** ~86%
- **Test Accuracy:** ~82.2%
- **Test Loss:** ~0.7035

![Training Accuracy and Loss](assets/training_curves.png)

## Usage

### 1. Training the Model

Run the Jupyter notebook:
```bash
jupyter notebook layan-classification-image.ipynb
```

Execute cells sequentially to:
- Load and preprocess the dataset
- Build the model architecture
- Train the classifier
- Save the trained model

### 2. Making Predictions with `predict.py`

The `predict.py` script provides a command-line interface for classifying flower images.

#### Basic Usage

```bash
python predict.py /path/to/image.jpg /path/to/saved_model.h5
```

#### Full Options

```bash
python predict.py image_path model_path --top_k 5 --category_names label_map.json --gpu
```

#### Examples

**Basic prediction with top 1 class:**
```bash
python predict.py test_images/wild_pansy.jpg saved_models/model.h5
```

**Top 5 predictions with category names:**
```bash
python predict.py test_images/wild_pansy.jpg saved_models/model.h5 --top_k 5 --category_names label_map.json
```

**Using GPU for faster inference:**
```bash
python predict.py test_images/wild_pansy.jpg saved_models/model.h5 --gpu
```

### Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `image_path` | str | Yes | Path to input image file |
| `model_path` | str | Yes | Path to saved Keras model (.h5) |
| `--top_k` | int | No | Number of top predictions to return (default: 1) |
| `--category_names` | str | No | Path to JSON file mapping labels to flower names |
| `--gpu` | flag | No | Enable GPU for inference |

### Expected Output

Without `--category_names`:
```
Top 1 predictions:
Class: 51, Probability: 0.9567
```

With `--category_names` and `--top_k 5`:
```
Top 5 predictions:
wild pansy: 0.9567
petunia: 0.0234
primula: 0.0089
oxeye daisy: 0.0056
buttercup: 0.0032
```

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~87% |
| Test Accuracy | 82.24% |
| Test Loss | 0.7035 |

### Strengths
- High accuracy on diverse flower species
- Fast inference (MobileNet architecture)
- Robust to different lighting conditions and angles

### Limitations
- May struggle with flowers that look very similar
- Performance decreases with very low-resolution images
- Requires GPU for reasonable training time

## Files Description

### `layan-classification-image.ipynb`
Main training notebook containing:
- Data loading and preprocessing
- Model architecture definition
- Training loop with progress visualization
- Model saving functionality

### `predict.py`
Command-line inference script with:
- Image loading and preprocessing
- Model loading with custom object scope
- Top-K prediction generation
- Optional label mapping to flower names

### `label_map.json`
JSON dictionary mapping numeric class indices (0-101) to flower names:
- Based on Oxford Flowers 102 dataset
- Contains 102 flower species names
- Used for human-readable predictions

### Saved Model File (`.h5`)
Keras HDF5 file containing:
- Model architecture
- Trained weights
- Optimizer state
- Training configuration

## Implementation Details

### Image Preprocessing for Inference

The `process_image()` function:
1. Converts image to tensor
2. Resizes to 224×224 pixels
3. Normalizes pixel values to [0, 1]
4. Returns processed numpy array

### Model Loading for Prediction

```python
with tf.keras.utils.custom_object_scope({'call_hub_layer': call_hub_layer}):
    model = tf.keras.models.load_model(model_path)
```

This ensures the custom `call_hub_layer` function is properly recognized when loading the model.

## Troubleshooting

### Common Issues and Solutions

1. **"module 'tensorflow' has no attribute 'hub'"**
   - Solution: `pip install tensorflow-hub`

2. **Out of Memory during training**
   - Reduce batch size (default: 32)
   - Use smaller MobileNet variant

3. **Model loading error with custom layer**
   - Ensure `call_hub_layer` is defined before loading
   - Use `custom_object_scope` as shown in predict.py

4. **Low validation accuracy**
   - Increase training epochs
   - Adjust dropout rate
   - Add data augmentation

## Acknowledgments

- **Udacity Academy** for project guidance
- **Oxford University** for the Flowers 102 dataset
- **TensorFlow Hub** for pre-trained MobileNet V3 model

## License

This project is for educational purposes as part of Udacity Academy's AI Nanodegree program.

---

**Author:** Layan Buirat  
**Date:** April–7 to 15 2026  
**Course:** Udacity Academy AI Nanodegree
