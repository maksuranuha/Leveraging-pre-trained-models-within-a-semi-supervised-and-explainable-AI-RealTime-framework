# Betel Leaf Disease Detection using Semi-Supervised Learning and Explainable AI

DOI: https://doi.org/10.1016/j.jafr.2025.102142

Journal: Journal of Agriculture and Food Research, Volume 22 (2025), Article 102142

## 📋 Overview

This repository presents an innovative approach to automated betel leaf disease detection by leveraging pre-trained deep learning models integrated with semi-supervised learning and explainable AI techniques. The research demonstrates a paradigm shift from traditional supervised-only methods to a hybrid framework that achieves state-of-the-art performance while minimizing reliance on extensively labeled datasets.

## 🎯 Project Objectives

The primary goals of this research are to:

- **Enhance Disease Classification**: Develop a reliable system to distinguish healthy betel leaves from diseased ones with high accuracy
- **Reduce Labeling Costs**: Implement semi-supervised approaches requiring only 30% labeled data while maintaining superior performance
- **Enable Interpretability**: Integrate explainable AI methods to visualize and understand disease detection decisions
- **Support Agricultural Communities**: Provide farmers and agricultural stakeholders with a practical real-time detection tool

## 📊 Key Results

| Approach | Model | Accuracy | Labeled Data Required |
|----------|-------|----------|----------------------|
| Supervised Learning | DenseNet-201 | 99.23% | 100% |
| Semi-Supervised Learning | FixMatch | 98% | 30% |
| Semi-Supervised Learning | MixMatch | 91.27% | 30% |
| Semi-Supervised Learning | MeanTeacher | 81% | 30% |

## Semi-Supervised Learning Impact

### Table : Several Label Data Implementations with Varied Ratio and Threshold Values

| Label Data (%) | Threshold | Validation Accuracy (%) | Test Accuracy (%) | Bacterial Leaf Disease (%) | Dried Leaf (%) | Fungal Brown Spot Disease (%) | Healthy Leaf (%) | Epoch | Training Time (min) |
|---|---|---|---|---|---|---|---|---|---|
| 30% | 0.75 | 91% | 98% | 100% | 100% | 100% | 93% | 25 | 16.41 |
| 20% | 0.75 | 88% | 96% | 100% | 100% | 95% | 93% | 25 | 10.49 |
| 10% | 0.75 | 56% | 57% | 100% | 100% | 33% | 21% | 13 | 5.18 |
| 5% | 0.75 | 48% | 41% | 100% | 9% | 14% | 21% | 17 | 6.51 |
| 30% | 0.80 | 89% | 96% | 100% | 100% | 95% | 93% | 25 | 16.54 |
| 30% | 0.85 | 87% | 94% | 100% | 100% | 91% | 89% | 25 | 16.62 |
| 30% | 0.90 | 79% | 76% | 100% | 100% | 43% | 71% | 9 | 6.01 |
| 40% | 0.75 | 91% | 98% | 100% | 100% | 95% | 93% | 25 | 19.48 |
| 50% | 0.75 | 94% | 98.8% | 100% | 100% | 100% | 96% | 25 | 30.12 |

**Key Findings:**
- 30% labeled data with 0.75 threshold achieves optimal 98% accuracy
- Higher thresholds (0.80-0.95) result in lower accuracy due to stricter pseudo-label filtering
- Lower labeled data percentages (5-10%) show significant accuracy drops
- 50% labeled data shows only marginal improvement (0.8%) over 30% with 83% more labeling effort
  
## 🎓 Methodology Pipeline

### Procedure Pipeline to Classify Betel Leaf Disease

The complete pipeline consists of:
1. Dataset collection and preprocessing
2. Data augmentation and splitting
3. Parallel supervised learning with three pre-trained models (InceptionV3, EfficientNet, DenseNet-201)
4. Semi-supervised learning integration (FixMatch, MixMatch, MeanTeacher)
5. Explainable AI visualization with DenseNet-201
6. Performance evaluation and model comparison
7. Real-time web application deployment

## 📋 Dataset Classes & Characteristics

### Betel Leaf Varieties
<img width="680" height="185" alt="Screenshot 2025-12-03 105728" src="https://github.com/user-attachments/assets/8f87b2aa-e752-4af8-9a9f-342f3a1b5d77" />

**Healthy Leaf**
- Vivid green color
- Smooth texture
- No yellow or brown spots
- No dark areas or holes

**Fungal Brown Spot Disease**
- Miniature asymmetric brown spots
- Yellowish halo around spots
- Gradual expansion from center to edges
- Potential leaf deformation

**Dried Leaf Disease**
- Yellow-brown hue
- Drying symptoms from edges inward
- Folded appearance
- May turn completely brown in extreme cases

**Bacterial Leaf Disease**
- Pale yellow discoloration
- Blackish dark patches
- Rotting symptoms
- Yellow haze around affected areas

## 📸 Data Augmentation Details
<img width="684" height="255" alt="image" src="https://github.com/user-attachments/assets/95a74ea0-feb0-41c8-995f-3f3a04054361" />

1. **Image Resizing**: Original 600×800 pixels → 224×224 pixels
2. **Data Augmentation** (Roboflow):
   - Horizontal flipping
   - Rotation and brightness adjustments
   - Contrast adjustments
   - Dataset expansion: 250 → 570+ images per class
3. **Data Splitting**: 80% training, 20% testing
4. **Normalization**: Pixel values scaled to [0,1] range

## 🏗️ Architecture Overview

### Supervised Learning Models

Three pre-trained convolutional neural networks were evaluated:

**DenseNet-201** (Best Performer)
- 201 layers with 19.6M parameters
- Dense connectivity enabling superior gradient flow
- Achieved 99.23% accuracy with 15 epochs
- Training time: 19.27 minutes

**InceptionV3**
- Google's multi-branch CNN architecture
- Asymmetric convolutional layers for feature extraction
- Achieved 92.72% accuracy
- Training time: 14.49 minutes

**EfficientNetV2**
- Optimized for speed-accuracy trade-offs
- Stochastic depth regularization
- Achieved 97.31% accuracy
- Training time: 17.43 minutes

### Semi-Supervised Learning Algorithms

**FixMatch** (Top Performer)
- Pseudo-labeling for weakly augmented unlabeled images
- High-confidence predictions from weak augmentations
- Consistency enforcement from strong augmentations
- Achieved 98% accuracy with only 30% labeled data
- Training time: 16.41 minutes

**MixMatch**
- Combines labeled and unlabeled data through consistency regularization
- MixUp data augmentation and entropy minimization
- Achieved 91.27% accuracy
- Training time: 8.94 minutes

**MeanTeacher**
- Teacher-student consistency architecture
- Exponential moving average for weight averaging
- Achieved 81% accuracy
- Training time: 10.24 minutes

### Explainable AI Techniques

Four XAI visualization methods were integrated with DenseNet-201:

- **SmoothGrad**: Gradient-based sensitivity visualization with noise averaging
- **Vanilla Saliency**: Feature importance heatmaps via gradient computation
- **GradCAM++**: Enhanced spatial resolution highlighting affected regions
- **Faster ScoreCAM**: Activation map weighting without gradient computation

## 💻 Technical Implementation

### Dependencies & Tools

| Category | Tools & Frameworks |
|----------|-------------------|
| Language | Python 3.x |
| Deep Learning | TensorFlow/Keras, PyTorch |
| Machine Learning | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Image Processing | OpenCV, skimage, PIL, torchvision |
| System Utilities | os, shutil, glob, psutil, tqdm |
| Web Framework | Flask, React |

### Hardware Configuration

- **Processor**: AMD Ryzen 5 5600G @ 3901 MHz (6 cores)
- **RAM**: 16 GB (2x8GB)
- **GPU**: NVIDIA GeForce GTX 1660 Ti
- **OS**: Microsoft Windows 11 Pro

### Training Configuration

**Supervised Models**
- Optimizer: Adam
- Learning Rate: 0.0009
- Batch Size: 10
- Input Size: 224×224 pixels
- Loss Function: Categorical Cross-Entropy
- Epochs: 25 (with early stopping, patience=5)
- Dropout: 0.6

**Semi-Supervised Models**
- Optimizer: Adam
- Learning Rate: 0.0001
- Batch Size: 8 (labeled), 32 (unlabeled)
- Label Data Ratio: 30%
- Threshold (FixMatch): 0.75
- Loss Function: Categorical Cross-Entropy

## 📈 Performance Metrics

### Evaluation Criteria

- **Accuracy**: Overall correct predictions across all classes
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall

### Table : Classification Report of Supervised Models

| Model | Class Name | Precision | Recall | F1-score | Accuracy |
|-------|-----------|-----------|--------|----------|----------|
| **DenseNet-201** | Bacterial Leaf | 1.00 | 0.98 | 0.99 | 0.98 |
| | Dried Leaf | 1.00 | 1.00 | 1.00 | 1.00 |
| | Fungal Brown Spot | 0.98 | 1.00 | 0.99 | 1.00 |
| | Healthy Leaf | 0.98 | 0.98 | 0.98 | 0.98 |
| | **Overall Accuracy** | | | | **99.23%** |
| **InceptionV3** | Bacterial Leaf | 0.80 | 1.00 | 0.88 | 1.00 |
| | Dried Leaf | 1.00 | 1.00 | 1.00 | 1.00 |
| | Fungal Brown Spot | 0.96 | 0.92 | 0.94 | 0.92 |
| | Healthy Leaf | 1.00 | 0.79 | 0.88 | 0.79 |
| | **Overall Accuracy** | | | | **92.72%** |
| **EfficientNetV2** | Bacterial Leaf | 0.95 | 0.98 | 0.97 | 0.98 |
| | Dried Leaf | 1.00 | 1.00 | 1.00 | 1.00 |
| | Fungal Brown Spot | 0.96 | 0.93 | 0.95 | 0.93 |
| | Healthy Leaf | 0.96 | 0.96 | 0.96 | 0.96 |
| | **Overall Accuracy** | | | | **97.31%** |

### Table : Classification Report of Semi-Supervised Models

| Model | Class Name | Precision | Recall | F1-score | Accuracy |
|-------|-----------|-----------|--------|----------|----------|
| **FixMatch** | Bacterial Leaf | 0.92 | 1.00 | 0.96 | 1.00 |
| | Dried Leaf | 1.00 | 1.00 | 1.00 | 1.00 |
| | Fungal Brown Spot | 1.00 | 1.00 | 1.00 | 1.00 |
| | Healthy Leaf | 1.00 | 0.93 | 0.96 | 0.92 |
| | **Overall Accuracy** | | | | **98%** |
| **MeanTeacher** | Bacterial Leaf | 0.68 | 0.95 | 0.79 | 0.95 |
| | Dried Leaf | 1.00 | 0.44 | 0.61 | 0.44 |
| | Fungal Brown Spot | 1.00 | 0.65 | 0.79 | 0.65 |
| | Healthy Leaf | 0.83 | 1.00 | 0.91 | 1.00 |
| | **Overall Accuracy** | | | | **81%** |
| **MixMatch** | Bacterial Leaf | 0.96 | 0.68 | 0.80 | 0.69 |
| | Dried Leaf | 1.00 | 1.00 | 1.00 | 1.00 |
| | Fungal Brown Spot | 0.97 | 0.89 | 0.93 | 0.90 |
| | Healthy Leaf | 0.69 | 0.97 | 0.81 | 0.97 |
| | **Overall Accuracy** | | | | **91.27%** |

## 📊 Model Execution Times

### Table : Supervised Model Evaluation with Execution Time

| Model Name | Execution Time (minutes) |
|-----------|--------------------------|
| DenseNet-201 | 19.27 |
| EfficientNetV2 | 17.43 |
| InceptionV3 | 14.49 |

### Semi-Supervised Models Outcome Evaluation with Execution Time

| Method | Execution Time (minutes) |
|--------|--------------------------|
| FixMatch | 16.41 |
| MixMatch | 8.94 |
| MeanTeacher | 10.24 |

## 🔄 Data Preprocessing Pipeline

### FixMatch Algorithm Augmentation Process

<img width="924" height="705" alt="Screenshot 2025-12-03 104958" src="https://github.com/user-attachments/assets/e285a330-7cd4-466c-bd2e-0999b91416f8" />

**FixMatch Algorithm Augmentation Types**

The research employs 16 distinct augmentation techniques:
- AutoContrast
- Equalize
- Invert
- Rotate
- Posterize
- Solarize
- Color
- Contrast
- Brightness
- Sharpness
- ShearX
- ShearY
- Cutout
- TranslateX
- TranslateY
- Identity

** A. Original vs Strong Augmentation**

Strong augmentation applies multiple aggressive transformations including posterization, color adjustments, histogram equalization, rotation, solarization, shearing, masking, and translation to enhance model robustness on unlabeled data.

** B. Original vs Weak Augmentation**

Weak augmentation applies only mild transformations including horizontal flipping, random rotation, and moderate shifting to maintain consistent predictions for pseudo-labeling.

## 🔍 Explainable AI Visualization

<img width="855" height="192" alt="Screenshot 2025-12-03 105121" src="https://github.com/user-attachments/assets/b9046302-ba34-4e4e-89f9-5b01ca799229" />

### XAI Methods Comparison for Disease Detection

This visualization demonstrates four XAI techniques applied to a diseased betel leaf sample:

**Original Image**: Shows affected leaf area with disease symptoms

**SmoothGrad**: Provides smooth gradient-based visualization by averaging multiple noise-affected samples, showing general disease regions with minimal noise

**Vanilla Saliency**: Generates detailed heatmap through gradient computation, appearing noisier and scattered across affected areas, providing fine-grained feature importance

**GradCAM++**: Enhanced class activation mapping offering clearer vision of detected disease regions with better spatial resolution

**Faster ScoreCAM**: Advanced version concentrating on the most confident abnormality regions, providing the most focused visualization of disease location

**Interpretation**: The progression from SmoothGrad (smooth, general) to Faster ScoreCAM (concentrated, precise) shows how different XAI techniques balance smoothness with precision in highlighting affected leaf areas.

## 🌟 Research Contributions

### Table : Numerical Comparison of Previous Studies and Our Study

| Studies | Dataset Size | Classes | Best Model | Accuracy |
|---------|--------------|---------|-----------|----------|
| [25] | 1,047 images | 6 | Extreme Learning Machine | 97.00% |
| [16] | 180 images | 3 | KNN | 100% |
| [20] | 10,662 images | 2 | BLCNN | 97.26% |
| [17] | 360 images | 3 | SVM | 100% |
| [18] | 1,275 images | 2 | SVM + GMM | 83.69% |
| **Our Proposed** | **1,000 images (Augmented: 2,589)** | **4** | **DenseNet-201 + FixMatch** | **99.23% + 98% (30% labeled)** |

**Key Advantages:**
- Only study combining supervised + semi-supervised + XAI approaches
- Achieved 99.23% accuracy with fully labeled data (DenseNet-201)
- Achieved 98% accuracy with only 30% labeled data (FixMatch)
- Minimal 1.23% accuracy difference between supervised and semi-supervised approaches
- Comprehensive 4-class disease categorization
- Real-time interpretable predictions through XAI integration
- Practical web application deployment

## 🚀 Future Directions

- Extend to multiple plant species and broader disease categories
- Incorporate time-series data for disease progression tracking
- Integrate ensemble-based semi-supervised methods
- Deploy quantization and pruning for edge device compatibility
- Develop personalized disease management systems based on crop health
- Expand dataset with additional disease patterns and environmental conditions

## ⚠️ Limitations

- Focus on single plant species (betel leaf)
- Region-specific dataset from Bangladesh
- Hyperparameter sensitivity (batch size, threshold values)
- Potential accuracy variations with different crop growth stages

## 📚 Key Methodology Highlights

The research employs a rigorous pipeline combining cutting-edge techniques:

1. Transfer learning from ImageNet pre-trained models
2. Semi-supervised learning for unlabeled data utilization
3. Multiple augmentation strategies (16 distinct types) for data enrichment
4. Four complementary XAI methods for model interpretability
5. Comprehensive performance evaluation across multiple metrics
6. Extensive hyperparameter optimization for both supervised and semi-supervised approaches

## 💡 Agricultural Impact

This framework offers transformative benefits for betel leaf cultivation:

- **Quality Control**: Automated detection of diseased leaves prevents fraudulent sales
- **Productivity**: Early disease identification enables timely intervention
- **Economic Growth**: Reduces crop losses and enhances farmer profitability
- **Employment**: Creates opportunities for agricultural technology specialists
- **Sustainability**: Supports food security and sustainable farming practices

## Application Use Cases

- Farmer disease diagnosis in cultivation fields
- Quality assurance in processing facilities
- Market inspection and trading oversight
- Agricultural research and disease monitoring
- Crop yield optimization through early intervention

---
