# 🗑️ Waste Classification Using CNN

## 📌 Project Overview
An intelligent waste classification system using Convolutional Neural Networks (CNN) to automatically categorize waste materials into 12 different classes. This project aims to facilitate proper waste segregation and promote efficient recycling practices.

---

## 🎯 Problem Statement
Improper waste segregation is a major environmental challenge leading to:
- Contamination of recyclable materials
- Increased landfill waste
- Environmental pollution
- Inefficient recycling processes

**Solution:** Develop a deep learning model that can automatically classify waste images into appropriate categories with high accuracy, enabling automated waste sorting systems.

---

## 📊 Dataset Information
- **Source:** [Kaggle - Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- **Total Images:** 15,515
- **Number of Classes:** 12
- **Image Format:** JPEG/PNG
- **Average Image Size:** 350 x 352 pixels

### Waste Classes:
1. Battery
2. Biological
3. Brown Glass
4. Cardboard
5. Clothes
6. Green Glass
7. Metal
8. Paper
9. Plastic
10. Shoes
11. Trash
12. White Glass

---

## 📈 Class Distribution Analysis
| Class | Number of Images | Percentage |
|-------|-----------------|------------|
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

**Imbalance Ratio:** 8.77x (Clothes vs Brown Glass)

---

## 🔍 Week 1 Milestone: Exploratory Data Analysis (EDA)

### Completed Tasks:
✅ Dataset acquisition and loading  
✅ Data structure analysis  
✅ Class distribution visualization  
✅ Image quality assessment  
✅ Duplicate detection (19 duplicates identified)  
✅ Statistical analysis  
✅ Data preprocessing strategy defined  

### Key Findings:
- **Total Images:** 15,515 (after cleanup)
- **Image Quality:** 100% valid images (0 corrupted)
- **Duplicates:** 19 images (0.12% - negligible impact)
- **Class Imbalance:** Significant (8.77x ratio)
- **Recommended Input Size:** 224x224 pixels
- **Data Split Strategy:** 70% Train, 15% Validation, 15% Test

### EDA Insights:
1. **Class Imbalance Detected:** Clothes class is overrepresented (5,325 images) while Brown Glass is underrepresented (607 images)
2. **Solution Strategy:** Implement data augmentation and class weighting during training
3. **Image Variability:** Images have varying dimensions (51px to 888px), requiring standardization
4. **Dataset Quality:** High-quality dataset with minimal corruption or duplication

---

## 🛠️ Technology Stack

### Programming Language:
- Python 3.12+

### Libraries & Frameworks:
- **Deep Learning:** TensorFlow 2.19.0, Keras
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Image Processing:** Pillow (PIL)
- **Model Development:** scikit-learn

### Platform:
- Google Colab (with GPU: T4)
- Kaggle Dataset API

---

## 📂 Project Structure
```
Week-1/
├── README.md                              # Project overview
├── PROBLEM_STATEMENT                      # Detailed problem description
├── DATASET_INFO                           # Dataset documentation
├── requirements                           # Python dependencies
└── notebooks/
    ├── Waste_Classification_EDA.ipynb     # Exploratory data analysis
    └── dataset_info.json                  # Dataset metadata

```

---

## 🚀 Getting Started

### Prerequisites:
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn pillow scikit-learn kagglehub
```

### Running the EDA Notebook:
1. Open Google Colab
2. Upload `Waste_Classification_EDA.ipynb`
3. Set Runtime to GPU (Runtime → Change runtime type → T4 GPU)
4. Run all cells sequentially

---

## 📊 EDA Results Summary

### Dataset Statistics:
- **Total Images:** 15,515
- **Classes:** 12
- **Average per Class:** 1,292.9 images
- **Largest Class:** Clothes (5,325 images)
- **Smallest Class:** Brown Glass (607 images)
- **Imbalance Ratio:** 8.77x

### Data Quality:
- **Corrupted Images:** 0
- **Duplicate Images:** 19 (0.12%)
- **Image Formats:** JPEG (99.8%), PNG (0.2%)
- **Image Dimensions:** Variable (51px to 888px)

### Recommendations:
1. ✅ Use data augmentation (rotation, flip, zoom, brightness adjustment)
2. ✅ Apply class weighting during training to handle imbalance
3. ✅ Standardize images to 224x224 pixels
4. ✅ Implement stratified train/validation/test split (70/15/15)
5. ✅ Use transfer learning (MobileNetV2 or similar) for better accuracy

---

## 🎯 Next Steps (Week 2)

### Planned Activities:
1. Build CNN model architecture using transfer learning
2. Implement data augmentation pipeline
3. Train model with class weighting
4. Evaluate model performance
5. Generate confusion matrix and classification report
6. Optimize hyperparameters

### Target Metrics:
- **Expected Accuracy:** 85-92%
- **Training Time:** 20-40 minutes (with GPU)
- **Model Size:** ~12 MB


