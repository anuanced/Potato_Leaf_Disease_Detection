# Potato Disease Classification: A Comparative Study of Deep Learning Models

## Abstract

This project presents a comparative analysis of two deep learning approaches for automated potato disease classification using leaf images. A custom Convolutional Neural Network (CNN) trained from scratch is evaluated against a MobileNetV2 model implemented using transfer learning. The system aims to study trade-offs between model complexity, inference time, and prediction confidence in an agricultural image classification context.

## 1. Introduction

Potato crops are susceptible to diseases such as Early Blight and Late Blight, which can significantly reduce yield if not detected at an early stage. Manual disease identification is time-consuming and requires expert knowledge. Automated image-based classification using deep learning offers a scalable solution for early diagnosis.

This project explores the effectiveness of two neural network architectures for potato disease detection and provides a web-based interface to compare their predictions in real time.

## 2. Objectives

The primary objectives of this project are:

* To develop a deep learning-based system for potato disease classification
* To compare a custom CNN model with a transfer learning approach
* To analyze inference time and prediction confidence of both models
* To demonstrate practical deployment of machine learning models using a web interface

## 3. System Overview

The system is implemented using a client–server architecture. The frontend provides a user interface for image upload, while the backend processes the image and returns predictions from both trained models via a REST API. The application runs locally and is intended for experimental and educational use.

## 4. Technologies Used

* **Programming Language:** Python
* **Backend Framework:** Flask
* **Frontend Technologies:** HTML, CSS, JavaScript
* **Machine Learning Framework:** TensorFlow and Keras
* **Visualization Library:** Chart.js
* **Execution Environment:** Local development setup

## 5. Project Structure

```
potato-disease-comparison/
│
├── backend/
│   ├── app.py
│   ├── utils.py
│   ├── requirements.txt
│   └── models/
│       ├── potatoes.h5
│       └── mobilenetv2_potato.h5
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── README.md
```

## 6. Model Description

### 6.1 Custom Convolutional Neural Network

The custom CNN is trained from scratch using labeled potato leaf images. It consists of multiple convolutional and pooling layers followed by fully connected layers for classification.

* Input size: 256 × 256 × 3
* Training approach: Supervised learning
* Key advantage: Lightweight architecture with faster inference
* Suitable for: Resource-constrained environments

### 6.2 MobileNetV2 (Transfer Learning)

MobileNetV2 is a pre-trained model originally trained on the ImageNet dataset. Transfer learning is applied by fine-tuning the top layers for potato disease classification.

* Input size: 224 × 224 × 3
* Training approach: Fine-tuning pre-trained weights
* Key advantage: Higher accuracy with limited training data
* Suitable for: Accuracy-critical applications

## 7. Image Preprocessing

The preprocessing pipeline differs for each model due to architectural requirements.

**Custom CNN:**

* Image resizing to 256 × 256
* Pixel normalization to the range [0, 1]

**MobileNetV2:**

* Image resizing to 224 × 224
* Preprocessing using MobileNetV2’s normalization method

## 8. Application Workflow

1. The user uploads a potato leaf image through the frontend interface
2. The image is sent to the backend via an HTTP POST request
3. Both models independently process the image
4. Predictions, confidence scores, and inference times are returned
5. Results are displayed for comparison

## 9. Output and Evaluation

Each model produces:

* Predicted disease class
* Confidence score for the prediction
* Inference time
* Probability distribution across all classes

The system also indicates whether both models agree on the predicted class, which can be used as a qualitative reliability indicator.

## 10. Limitations

* The system supports a limited number of disease classes
* No user authentication or dataset expansion features are included
* Performance is dependent on image quality and dataset diversity
* The application is designed for local execution only

## 11. Future Scope

Potential enhancements include:

* Addition of more crop diseases and plant types
* Integration of real-time camera-based prediction
* Deployment to cloud platforms
* Inclusion of quantitative evaluation metrics such as accuracy and F1-score
* Development of a mobile-friendly or standalone application

## 12. Conclusion

This project demonstrates the practical application of deep learning techniques in agricultural disease detection. By comparing a custom CNN with a transfer learning model, the study highlights performance trade-offs relevant to real-world deployment scenarios. The system serves as a foundation for further research in precision agriculture and intelligent decision-support tools.

## 13. Author

**Anusha Thosar**

* Undergraduate Student
* Field of Study: Electronics and Computer Engineering
* This project was developed as part of academic coursework and independent study, focusing on deep learning–based image classification, model performance comparison, and the application of machine learning techniques to agricultural problem domains.
