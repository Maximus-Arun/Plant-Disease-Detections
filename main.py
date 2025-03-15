import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os
from PIL import Image

# Function to download the model from Google Drive
def download_model():
    file_id = "1ITgWByH94WdOvEoF_l3m8HhjFDuthTAi"  
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url, stream=True)

    with open("trained.h5", "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

# Check if model is already downloaded, if not, download it
MODEL_PATH = "trained.h5"
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model... Please wait ⏳")
    download_model()
    st.write("Download complete! ✅")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Model Prediction Function
def model_prediction(test_image):
    image = Image.open(test_image)  # 🔹 Open uploaded image
    image = image.resize((128, 128))  # 🔹 Resize for model input
    input_arr = np.array(image) / 255.0  # 🔹 Normalize pixel values
    input_arr = np.expand_dims(input_arr, axis=0)  # 🔹 Expand dims for batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Streamlit Sidebar
st.sidebar.title("Dashboard")

# Streamlit Main UI
st.header("🌱 Crop Disease Recognition")
test_image = st.file_uploader("📷 Upload an Image:")

if test_image:
    st.image(test_image, use_column_width=True)

if st.button("🔍 Predict"):
    st.snow()
    result_index = model_prediction(test_image)

    # Class Names
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
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]

    # Display Prediction
    st.success(f"🌾 The model predicts: **{class_name[result_index]}**")
