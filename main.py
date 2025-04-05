import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

# Set page configuration
st.set_page_config(page_title="CropDoc: Disease Recognition", page_icon="🌿", layout="wide")

# Function to download the model from Google Drive
def download_model():
    file_id = "145PdUPkRDiH6FToezUaM8RlQGPfjC1QB"
    model_path = "trained_model.h5"
    if not os.path.exists(model_path):  # Check if the model already exists
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return model_path

# Model Prediction
def model_prediction(test_image):
    MODEL_PATH = download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0  # Normalize the image
    
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    
    return result_index

# Sidebar
st.sidebar.title("🌿 CropDoc Dashboard")

# About Section at the Bottom of Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.write("This project is developed by:")
st.sidebar.markdown("- **Arun S**  \n- **Dravidan A C**  \n- **Gowtham D**")
st.sidebar.write("A deep learning-based system for crop disease detection.")

# Main Title
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>Crop Disease Recognition 🌾</h1>", unsafe_allow_html=True)
st.write("### Upload an image of a crop leaf to identify the disease.")

# Image Upload
test_image = st.file_uploader("📸 Upload an Image", type=["jpg", "jpeg", "png"])

if test_image:
    st.image(test_image, caption="Uploaded Image", use_column_width=True)

# Predict Button
if st.button("🔍 Predict"):
    with st.spinner("Analyzing... Please wait."):
        progress_bar = st.progress(0)
        
        for percent in range(100):
            progress_bar.progress(percent + 1)
        
        result_index = model_prediction(test_image)
        progress_bar.empty()
    
    # Labels
    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']
    
    st.success(f"✅ **Prediction:** {class_names[result_index]}")
