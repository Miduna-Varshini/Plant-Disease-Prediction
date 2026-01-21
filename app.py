import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Prediction",
    layout="centered"
)

st.title("🌿 Plant Disease Prediction")
st.write("Upload a leaf image to predict the disease")

# -----------------------------
# GOOGLE DRIVE MODEL DOWNLOAD
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID_HERE"
MODEL_PATH = "plant_disease_cnn.h5"

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()

# -----------------------------
# CLASS NAMES
# ⚠️ MUST MATCH TRAINING ORDER
# -----------------------------
class_names = [
    "Apple___Apple_scab",
    "Apple___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Tomato___Late_blight"
]

IMG_SIZE = (128, 128)

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    predicted_class = class_names[predicted_index]

    st.success(f"🌱 Predicted Disease: **{predicted_class}**")
    st.info(f"🔍 Confidence: **{confidence:.2f}%**")
