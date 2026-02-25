# Hand Gesture Recognition Using Machine Learning

Engineering thesis project focused on automatic hand gesture recognition using computer vision and machine learning techniques.

---

## 🎓 Thesis Title

**"Hand Gesture Recognition Using Machine Learning"**

---

## 📌 Project Overview

The objective of this project was to develop a system capable of recognizing hand gestures representing American Sign Language (ASL) letters using machine learning algorithms.

The system pipeline includes:

1. Hand detection  
2. Landmark extraction  
3. Feature engineering  
4. Classical machine learning classification  
5. Deep learning experiments  
6. Inference on new images  

---

## 📊 Dataset

The project is based on the:

**American Sign Language Dataset (ASL)**  
Available on Kaggle:  
https://www.kaggle.com/datasets/ayuraj/asl-dataset

Due to dataset size limitations, the full dataset is not included in this repository.

A small example subset is available in:


data/DataSample/


This folder contains several sample images for demonstration purposes.

---

## 🖐 Hand Landmark Detection

For hand detection and landmark extraction, the project uses:

**MediaPipe Hands**

The library enables extraction of 21 hand landmarks per detected hand, which are then used as structured features for machine learning classifiers.

The landmark-based approach significantly reduces dimensionality compared to raw image input and improves classification efficiency.

---

## ⚙️ Technologies Used

- Python  
- OpenCV  
- MediaPipe (MediaPipe Hands)  
- Scikit-learn  
- AutoGluon  
- Convolutional Neural Networks (CNN)  

---

## 📈 Project Type

Engineering Thesis  
Computer Vision & Machine Learning  

---

## 👨‍💻 Author

Jakub Jarczyk
