# 🗑️ Waste Classification Using CNN - Complete Project

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0-red.svg)](https://keras.io/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

---

## 📌 Project Overview

An AI-powered waste classification system using Convolutional Neural Networks (CNN) with Transfer Learning to automatically categorize waste materials into **12 different classes**. This project demonstrates the complete machine learning pipeline from data exploration to model deployment, addressing real-world challenges in automated waste management.

### 🎯 Project Goal
Build a deep learning model for waste image classification to enable automated waste sorting systems for efficient waste management and recycling.

---

## 🚀 Project Status

| Week | Milestone | Status |
|------|-----------|--------|
| **Week 1** | Exploratory Data Analysis & Preprocessing | ✅ Complete |
| **Week 2** | CNN Model Training & Evaluation | ✅ Complete |
| **Week 3** | Testing, Documentation & Final Submission | ✅ Complete |

**Overall Status:** ✅ **PROJECT COMPLETE**

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Total Images** | 15,515 |
| **Waste Classes** | 12 |
| **Training Accuracy** | 93.8% |
| **Validation Accuracy** | 93.5% |
| **Test Accuracy** | 93.43% |
| **Test Loss** | 0.3608 |
| **Model Size** | 11.64 MB |
| **Training Time** | ~40 minutes |
| **Training Platform** | Google Colab (T4 GPU) |

---

## 🗑️ Waste Categories (12 Classes)

The system successfully classifies waste into the following categories:

1. 🔋 **Battery** - Hazardous waste
2. 🍃 **Biological** - Organic waste
3. 🟤 **Brown Glass** - Recyclable glass
4. 📦 **Cardboard** - Recyclable paper
5. 👕 **Clothes** - Textile waste
6. 🟢 **Green Glass** - Recyclable glass
7. 🔩 **Metal** - Recyclable metal
8. 📄 **Paper** - Recyclable paper
9. ♻️ **Plastic** - Recyclable plastic
10. 👟 **Shoes** - Textile waste
11. 🗑️ **Trash** - General waste
12. ⚪ **White Glass** - Recyclable glass

---

## 📂 Project Structure

```
Waste-Classification/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── PROBLEM_STATEMENT.md                # Problem description
│
├── Waste_Classification.ipynb          # Complete source code
│
├── best_waste_classifier.h5           # Trained model
├── training_history.png               # Training graphs
├── confusion_matrix.png               # Performance matrix
├── classification_report.txt          # Detailed metrics
├── sample_predictions.png             # Test predictions
│
└── Waste_Classification_Source_Code.zip  # Zipped package
```

---

## 🛠️ Technology Stack

### Programming & Frameworks:
- **Python:** 3.12+
- **Deep Learning:** TensorFlow 2.19.0, Keras 3.0
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Image Processing:** Pillow (PIL)
- **ML Tools:** scikit-learn

### Platform & Hardware:
- **Development Platform:** Google Colab
- **GPU:** NVIDIA T4
- **Dataset Source:** Kaggle

---

## 📊 Dataset Information

### Source:
**Kaggle:** [Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)  
**Author:** Mostafa Abla

### Statistics:
- **Total Images:** 15,515
- **Classes:** 12
- **Format:** JPEG (99.8%), PNG (0.2%)
- **Average Size:** 350 × 352 pixels
- **Data Quality:** 100% valid images

### Class Distribution:

| Class | Images | Percentage |
|-------|--------|------------|
| Clothes | 5,325 | 34.3% |
| Shoes | 1,977 | 12.7% |
| Paper | 1,050 | 6.8% |
| Biological | 985 | 6.3% |
| Battery | 945 | 6.1% |
| Cardboard | 891 | 5.7% |
| Plastic | 865 | 5.6% |
| White Glass | 775 | 5.0% |
| Metal | 769 | 5.0% |
| Trash | 697 | 4.5% |
| Green Glass | 629 | 4.1% |
| Brown Glass | 607 | 3.9% |

**Class Imbalance:** 8.77× ratio (Clothes vs Brown Glass)

---

## 🏗️ Model Architecture

### Base Model: MobileNetV2
- **Pre-trained on:** ImageNet
- **Input Shape:** 224 × 224 × 3
- **Weights:** Frozen during training

### Custom Classification Layers:
```
Input: 224×224×3 RGB Image
    ↓
MobileNetV2 (Frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (50%)
    ↓
Dense (512, ReLU)
    ↓
BatchNormalization
    ↓
Dropout (50%)
    ↓
Dense (256, ReLU)
    ↓
BatchNormalization
    ↓
Dropout (30%)
    ↓
Dense (12, Softmax)
    ↓
Output: 12 class probabilities
```

### Model Parameters:
- **Total Parameters:** 3,051,340
- **Trainable Parameters:** 791,820
- **Non-trainable Parameters:** 2,259,520
- **Model Size:** 11.64 MB

---

## 🎯 Training Configuration

### Hyperparameters:
- **Image Size:** 224 × 224 pixels
- **Batch Size:** 32
- **Epochs:** 50
- **Learning Rate:** 0.0001
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy

### Data Split:
- **Training Set:** 10,860 images (70%)
- **Validation Set:** 2,327 images (15%)
- **Test Set:** 2,328 images (15%)
- **Split Method:** Stratified

### Data Augmentation:
- Rotation: ±40 degrees
- Horizontal/Vertical flip
- Width/Height shift: ±20%
- Zoom range: ±20%
- Shear range: ±20%
- Brightness: 80-120%

### Class Weighting:
Applied to handle 8.77× class imbalance with calculated weights for each class.

---

## 📈 Model Performance

### Results:
- **Training Accuracy:** 93.8%
- **Validation Accuracy:** 93.5%
- **Test Accuracy:** 93.43%
- **Test Loss:** 0.3608

### Analysis:
The model achieved excellent performance across all metrics, successfully handling:
- Severe class imbalance (8.77× ratio) through class weighting
- Visual similarity between classes using transfer learning
- Multi-class classification with 12 distinct waste categories

The high accuracy (93.43%) demonstrates the effectiveness of:
- Transfer learning with MobileNetV2
- Strategic data augmentation
- Proper handling of class imbalance
- Well-designed architecture with regularization

---

## 🚀 Getting Started

### Prerequisites:
```bash
pip install -r requirements.txt
```

### Quick Start - Load Model:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load the trained model
model = keras.models.load_model('best_waste_classifier.h5')

# Define class names
classes = ['battery', 'biological', 'brown-glass', 'cardboard', 
           'clothes', 'green-glass', 'metal', 'paper', 
           'plastic', 'shoes', 'trash', 'white-glass']

# Function to classify waste
def classify_waste(image_path):
    """Classify a waste image"""
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = classes[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    
    return predicted_class, confidence

# Example usage
waste_type, conf = classify_waste('your_image.jpg')
print(f"Predicted: {waste_type} (Confidence: {conf:.2f}%)")
```

---

## 📊 Project Milestones

### ✅ Week 1: Exploratory Data Analysis

**Accomplishments:**
- Downloaded and explored dataset (15,515 images)
- Performed comprehensive EDA
- Analyzed class distribution
- Identified 8.77× class imbalance
- Detected duplicate images (0.12%)
- Assessed image quality

**Deliverables:**
- EDA Notebook
- Dataset documentation
- Visualizations

---

### ✅ Week 2: Model Training & Evaluation

**Accomplishments:**
- Built CNN architecture with MobileNetV2
- Implemented data augmentation
- Applied class weighting
- Trained model for 50 epochs
- Generated evaluation metrics

**Deliverables:**
- Trained model (best_waste_classifier.h5)
- Training visualizations
- Confusion matrix
- Classification report

---

### ✅ Week 3: Testing & Documentation

**Accomplishments:**
- Tested model on validation set
- Created complete project notebook
- Generated comprehensive documentation
- Prepared final presentation

**Deliverables:**
- Complete source code
- Sample predictions
- Usage guide
- Final presentation

---

## 🎯 Key Learning Outcomes

### Technical Skills Gained:
✅ Deep Learning with TensorFlow/Keras  
✅ Transfer Learning implementation  
✅ Handling imbalanced datasets  
✅ Data augmentation techniques  
✅ Model evaluation metrics  
✅ End-to-end ML pipeline development  

### Understanding Acquired:
✅ Challenges in multi-class classification  
✅ Impact of class imbalance on model performance  
✅ Importance of data preprocessing  
✅ CNN architecture design  
✅ Model training optimization  

---

## 💡 Key Success Factors

### Factor 1: Transfer Learning
**Approach:** Used MobileNetV2 pre-trained on ImageNet  
**Impact:** Leveraged powerful feature extraction capabilities  
**Result:** Achieved 93.43% accuracy on complex 12-class problem

### Factor 2: Class Imbalance Handling
**Approach:** Applied class weighting and strategic augmentation  
**Impact:** Balanced learning across all classes  
**Result:** Consistent performance across minority classes

### Factor 3: Architecture Design
**Approach:** Custom classification head with dropout and batch normalization  
**Impact:** Prevented overfitting while maintaining capacity  
**Result:** Strong generalization (93.5% validation accuracy)

---

## 🌟 Model Strengths

### What the Model Does Well:
- **High Accuracy:** 93.43% on diverse waste categories
- **Robust Classification:** Handles visual similarity between classes
- **Balanced Performance:** Works well across all 12 classes
- **Efficient Size:** Only 11.64 MB for easy deployment
- **Fast Inference:** Quick predictions suitable for real-time use

### Real-World Readiness:
- Trained on real-world dataset with natural variations
- Handles different lighting conditions through augmentation
- Robust to image quality variations
- Ready for deployment in automated systems

---

## 🌍 Real-World Applications

This model is ready for deployment in:

- **Automated Waste Sorting:** Smart bins in public places with 93%+ accuracy
- **Recycling Centers:** Automated segregation systems for efficient processing
- **Environmental Monitoring:** Waste analytics and tracking systems
- **Educational Tools:** Interactive waste classification learning
- **Smart Cities:** IoT-enabled waste management infrastructure
- **Mobile Applications:** On-device waste classification for users

---

## 🚀 Future Development

### Short-term Goals:
- Deploy as REST API for web integration
- Create mobile application for end-users
- Optimize model for edge devices
- Add confidence thresholding for uncertain predictions

### Long-term Vision:
- Real-time video processing for conveyor belts
- Multi-label classification for mixed waste
- Integration with IoT smart bins
- Expand to more waste categories
- Develop region-specific models

---

## 📚 Documentation Files

### Project Documentation:
- Complete Project Notebook (.ipynb)
- Problem Statement
- Dataset Information
- Training Report
- Classification Report
- Usage Guide

---

## 👨‍💻 Author

**Neeta T P**  
AICTE Machine Learning Internship

### Contact:
- 📧 Email: neeta.tp18@gmail.com
- 💻 GitHub: https://github.com/NEETA-TP/Waste-Classification

---

## 🙏 Acknowledgments

### Special Thanks:
- **AICTE Internship Program** - For the learning opportunity
- **Mostafa Abla** - For the Kaggle dataset
- **TensorFlow Team** - For the deep learning framework
- **Google Colab** - For free GPU resources

### References:
1. MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler et al., 2018)
2. Kaggle Dataset: Garbage Classification
3. TensorFlow Documentation
4. Deep Learning best practices

---

## 📝 Project Reflection

This project successfully demonstrated:
- Complete machine learning pipeline from data to deployment
- Effective handling of real-world challenges (class imbalance, visual similarity)
- High-accuracy multi-class image classification (93.43%)
- Production-ready model suitable for real-world deployment

The outstanding results (93.43% accuracy) showcase:
- The power of transfer learning for complex classification tasks
- Importance of proper data preprocessing and augmentation
- Effective strategies for handling imbalanced datasets
- Value of well-designed architecture with appropriate regularization

---

## 📊 Technical Specifications

### Development Environment:
- Platform: Google Colab
- GPU: NVIDIA T4
- Runtime: Python 3.12
- Framework: TensorFlow 2.19.0

### Model Specifications:
- Architecture: CNN with Transfer Learning
- Base Model: MobileNetV2
- Input: 224×224×3 RGB images
- Output: 12 class probabilities
- Size: 11.64 MB
- Accuracy: 93.43%

---

## 🎓 Learning Resources

### Recommended Resources:
- TensorFlow Tutorials
- Keras Documentation
- Deep Learning Specialization (Coursera)
- Fast.ai Practical Deep Learning
- Stanford CS231n

---

## 📧 Support & Contact

### Need Help?
- Email: neeta.tp18@gmail.com
- GitHub: https://github.com/NEETA-TP/Waste-Classification

### Project Repository:
https://github.com/NEETA-TP/Waste-Classification

---

## 🎉 Project Completion

This project represents a successful journey through:
- Data exploration and analysis
- Model development and training
- Achieving 93.43% test accuracy
- Documentation and presentation
- Ready for real-world deployment

**Thank you for exploring this waste classification project!**

---

**Project Status:** Complete ✅  
**Last Updated:** November 17, 2025  
**Version:** 1.0.0

---

**Made by Neeta T P | AICTE ML Internship 2025**

*"From learning to excellence - 93.43% accuracy achieved!"* 🌟

---
