import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Page config
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱", layout="centered")

# Model file name
MODEL_PATH = "trained.h5"

# Function to download model from Google Drive
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        file_id = "1nMmb9wrirOb7UgXc8YaORBDPm2_n_u84"  # Google Drive file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Model Prediction
def model_prediction(test_image, model):
    image = Image.open(test_image).convert("RGB")
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("🌿 Plant Health Dashboard")
st.sidebar.markdown("Upload a leaf image to check for plant diseases.")

# Main Title
st.title("🌱 Plant Disease Recognition")
st.markdown("This app predicts the type of **plant disease** based on the uploaded leaf image.")

# Image Upload
test_image = st.file_uploader("📤 Upload a leaf image (JPG or PNG):", type=["jpg", "jpeg", "png"])

if test_image:
    st.image(test_image, use_column_width=True, caption="📷 Uploaded Image")

    if st.button("🔍 Predict Disease"):
        st.markdown("⏳ Loading model and analyzing image...")
        model = load_model()
        result_index = model_prediction(test_image, model)

        # Class labels
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]

        predicted_disease = class_name[result_index]

        st.success(f"✅ **Prediction Result:** `{predicted_disease}`")
        

else:
    st.info("👈 Upload a plant leaf image to get started.")
