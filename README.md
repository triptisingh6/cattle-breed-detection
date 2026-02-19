## Two-Stage Cattle Breed Detection System

A research-oriented deep learning pipeline for automatic cattle breed classification using a two-stage architecture:

1. **YOLOv8** for cow detection  
2. **EfficientNet-B0** for breed classification  
3. **Grad-CAM** for explainability  

---

## Project Overview

This system takes an input image, detects the cow using YOLOv8, crops the largest detected cow, and classifies the breed using a fine-tuned EfficientNet-B0 model.

The pipeline improves classification robustness by focusing only on the detected animal region instead of the full image.

**Current Validation Accuracy: 87%**

---

## Architecture Pipeline

Input Image
↓
YOLOv8 (Cow Detection)
↓
Crop Largest Bounding Box
↓
EfficientNet-B0 (Breed Classification)
↓
Softmax Confidence
↓
Grad-CAM Visualization

---

## Dataset

- 5 Cattle Breeds:
  - Ayrshire
  - BrownSwiss
  - HolsteinFriesian
  - Jersey
  - RedDane

- ~400 images per breed
- Automatic cropping using YOLOv8
- 80/20 Train–Validation split
- Class imbalance handled using Focal Loss

---

## Model Details

### Detection Model
- Model: YOLOv8n (Ultralytics)
- Class used: `cow`
- Confidence threshold: 0.25

### Classification Model
- Backbone: EfficientNet-B0 (ImageNet pretrained)
- Transfer Learning with frozen backbone (first 5 epochs)
- Gradual unfreezing
- Loss Function: **Focal Loss**
- Optimizer: Adam (lr = 0.0005)
- Scheduler: ReduceLROnPlateau
- Early Stopping enabled
- Test-Time Augmentation (Horizontal Flip)

---

## Evaluation Metrics

- Validation Accuracy: **87%**
- Macro F1 Score: (fill after final run)
- Weighted F1 Score: (fill after final run)
- Confusion Matrix visualization included
- Sample prediction visualization included

---

## Explainability

Grad-CAM is used to visualize which regions influenced the breed prediction.

This improves:
- Model interpretability
- Research credibility
- Trust in predictions

---

## Project Structure

cattle-breed-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│ └── breed_classifier_best.pth
│
├── src/
│ ├── train.py
│ ├── dataset.py
│ ├── model.py
│ ├── inference.py
│ └── utils.py
│
├── app/
│ └── gradio_app.py
│
└── data/

---

## 🚀 Installation

Clone the repository:

git clone 
cd cattle-breed-detection

Install dependencies:

pip install -r requirements.txt

---

## Run the Gradio App
python app/gradio_app.py

The web interface will open in your browser.

---

## Future Improvements

Upgrade to YOLOv8m for better detection accuracy

Try EfficientNet-B2 or ConvNeXt

Multi-cow classification support

Full pipeline deployment on HuggingFace Spaces

Expand dataset to 10+ breeds

---

## Applications

Smart livestock management

Automated farm monitoring

Agricultural AI research

Computer Vision academic projects

---

## License

This project is for academic and research purposes.

---

## Author

Tripti Singh
BTech CSE
Deep Learning & Computer Vision Enthusiast