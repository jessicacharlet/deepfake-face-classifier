# deepfake-face-classifier
A CNN-based project to detect real vs. fake (deepfake) human face images. It uses a labeled dataset, trains a model, and predicts whether a face is real or fake.

# 🧠 Fake Face Detection using CNN

A simple deep learning project to detect real vs. fake (deepfake) face images using Convolutional Neural Networks (CNN) in Python with TensorFlow/Keras.

---

## 📁 Dataset

- **Name**: Real and Fake Face Detection
- **Source**: [Kaggle](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
- Contains two folders:
  - `training_real/` – images of real faces
  - `training_fake/` – deepfake (fake) images

---

## ⚙️ Technologies Used

- Python
- Google Colab
- TensorFlow / Keras
- OpenCV
- Matplotlib, Seaborn
- scikit-learn

---

## 🚀 How it Works

1. Load and preprocess real & fake face images
2. Normalize and resize images to 128x128
3. Build a CNN model using Keras
4. Train and validate the model
5. Evaluate performance using:
   - Accuracy
   - Confusion matrix
   - Classification report
6. Test on new custom images

---

## 📊 Results

- Final Accuracy: ~58% (basic model)
- Model can classify a face as **REAL** or **FAKE**
- Performance can be improved with more data and advanced models (e.g., transfer learning)

---

## 🖼️ Predict New Images

You can test the model with your own images using:
```python
model.predict(new_image)
