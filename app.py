import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image

# Load model and class names
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('animal_detector_model.keras')
    return model

@st.cache_data
def load_class_names():
    return np.load('class_names.npy')

model = load_model()
class_names = load_class_names()

# Prediction Function
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index]

    return predicted_label, confidence

# Streamlit UI
st.sidebar.title('Animal Species Classifier')
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Animal Recognition'])

# Optional splash image
if os.path.exists('animal_cover.png'):
    st.image('animal_cover.png', use_container_width=True)

if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Animal Detection Using Deep Learning</h1>", unsafe_allow_html=True)
    st.write("Upload an image of an animal and the model will tell you what it is!")

elif app_mode == 'Animal Recognition':
    st.header("Upload an Animal Image for Recognition")
    uploaded_image = st.file_uploader("Choose an Image")

    if uploaded_image is not None:
        file_path = os.path.join("temp_uploaded_image.jpg")
        with open(file_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        if st.button("Show Image"):
            st.image(uploaded_image, use_container_width=True)

        if st.button("Predict"):
            label, confidence = predict_image(file_path)
            st.subheader("Prediction:")
            st.success(f"Model is predicting: **{label}** ({confidence * 100:.2f}% confidence)")
