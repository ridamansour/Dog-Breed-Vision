# 🐕 End-to-End Dog Breed Classification with Deep Learning

This project builds a complete deep learning pipeline for classifying dog breeds from images. It uses modern computer vision techniques with TensorFlow and Keras, leveraging transfer learning for high performance on relatively limited datasets.

---

## 🔍 Project Overview

* **Goal**: Classify images of dogs into their respective breeds.
* **Dataset**: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) (or similar subset)
* **Approach**: End-to-end pipeline using:

  * Image preprocessing and augmentation
  * Transfer learning with pre-trained models (e.g., EfficientNet, ResNet)
  * Fine-tuning and model evaluation

---

## 🧱 Pipeline Steps

1. **Data Loading**

   * Loads images and labels
   * Organizes into train/test/validation sets

2. **Preprocessing**

   * Resizing, rescaling, normalization
   * Augmentation: flip, rotate, zoom, etc.

3. **Model Architecture**

   * Transfer learning base: e.g., `EfficientNetB0`
   * Custom top layers for classification
   * Fine-tuning for improved performance

4. **Training**

   * Uses callbacks: EarlyStopping, ModelCheckpoint
   * Metrics: Accuracy, Precision, Recall

5. **Evaluation**

   * Confusion matrix
   * Classification report
   * Accuracy curves (training vs. validation)

6. **Prediction & Inference**

   * Run predictions on custom dog images
   * Display top-k predicted breeds with confidence scores

---

## 📁 Project Structure

```
├── end_to_end_dog_vision.ipynb  # Main notebook with full pipeline
├── data/
│   ├── train/                   # Training images
│   ├── valid/                   # Validation images
│   └── test/                    # Test images
├── models/
│   └── dog_classifier.h5        # Saved model (optional)
├── utils/
│   └── preprocessing.py         # Custom data functions (if used)
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies
```

---

## 📦 Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ridamansour/dog-vision.git
   cd dog-vision
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:

   ```bash
   jupyter notebook end_to_end_dog_vision.ipynb
   ```

4. **Download dataset**:
   Ensure the dog image dataset is downloaded and extracted to the appropriate `/data` folder structure.
   
---

## 🔮 Future Work

* Improve model generalization with additional data
* Use data generators for large datasets
* Deploy as a web or mobile app using TensorFlow Lite or Streamlit
* Add Grad-CAM for model explainability

---

## 📝 License

Open source.
