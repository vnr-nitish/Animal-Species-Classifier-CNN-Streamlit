# 🐾 Multi-Class Animal Recognition Using CNN + Streamlit  
**Author:** Nitish Raj Vinnakota | 🔗 [LinkedIn](https://linkedin.com/in/vnr-nitish)  

---

## 🔍 Project Overview

This project implements a **deep learning-based animal classification system** that can recognize multiple species from images. The trained Convolutional Neural Network (CNN) is deployed through a **Streamlit web application**, allowing users to upload an image and receive a high-confidence prediction instantly.

---

## 🎯 Objective

> Build and deploy a multi-class image classification model to accurately identify animal species from input images using CNN and provide confidence scores via a user-friendly interface.

---

## 📁 Dataset Summary

- **Image Classes**: Multiple animal categories (e.g., Lion, Elephant, Tiger, Panda, etc.)  
- **Input Size**: 224 × 224 × 3 RGB Images  
- **Preprocessing**: Normalization, resizing, label encoding
- **Dataset Link:** https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

---

## 🧠 Model Architecture

- **Framework**: TensorFlow / Keras  
- **Type**: CNN  
- **Layers**:
  - Conv2D → MaxPooling → Flatten → Dense  
- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Output**: Class probabilities for each animal species

---

## ⚙️ Streamlit App Features

- 📤 Upload an image of any animal  
- ⚙️ Model processes and predicts the species  
- ✅ Displays the class name and **confidence score (%)**  
- 🖼️ Image preview before prediction

---

## 🧰 Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Streamlit  
- PIL

---

## 🚀 How to Run the App Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Animal-Species-Classifier-CNN-Streamlit

# 2. Navigate to the directory
cd Animal-Species-Classifier-CNN-Streamlit

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place model and class label files in the same directory
#    - animal_detector_model.keras
#    - class_names.npy

# 5. Run the Streamlit app
streamlit run app.py
```
---

## 🚀 Future Improvements

- 🌐 Deploy the app using **Streamlit Cloud**, **Render**, or **Hugging Face Spaces**
- 🧠 Add Grad-CAM visualizations for explainability of predictions
- 📱 Convert the model to **TensorFlow Lite (TFLite)** for mobile applications
- 🐍 Extend the app to detect **multiple animals in a single image** using object detection
- 🎯 Fine-tune the model with a larger, balanced animal image dataset

---

## 📫 Contact

For collaborations, feedback, or questions, feel free to reach out:

- 📧 **Email:** nvinnako2@gitam.in  
- 🔗 **LinkedIn:** [linkedin.com/in/vnr-nitish](https://linkedin.com/in/vnr-nitish)

---

> *"Empowering wildlife understanding with AI — one prediction at a time."*
