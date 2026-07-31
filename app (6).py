
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Image Classification App", layout="centered")
st.title("Image Classification: Decks vs. Walls")
st.write("Upload an image to classify it as 'Decks' or 'Walls' using two trained models.")

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
class_names = ['Decks', 'Walls']
CNN_MODEL_PATH = 'models/custom_cnn.keras'
TL_MODEL_PATH = 'models/mobilenetv3_transfer.keras'

@st.cache_resource
def load_models():
    st.spinner("Loading models...")
    try:
        if not os.path.exists(CNN_MODEL_PATH):
            st.error(f"Model file not found: {CNN_MODEL_PATH}")
            return None, None
        if not os.path.exists(TL_MODEL_PATH):
            st.error(f"Model file not found: {TL_MODEL_PATH}")
            return None, None
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
        tl_model = tf.keras.models.load_model(TL_MODEL_PATH)
        st.success("Models loaded successfully!")
        return cnn_model, tl_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

cnn_model, tl_model = load_models()

if cnn_model is None or tl_model is None:
    st.stop()

uploaded_file = st.file_uploader("Choose an image for classification...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_resized = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    st.subheader("Prediction Results:")

    models_to_predict = {'Custom CNN': cnn_model, 'MobileNetV3 TL': tl_model}

    for name, model in models_to_predict.items():
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_class_index = np.argmax(predictions)
        predicted_label = class_names[predicted_class_index]
        confidence = predictions[predicted_class_index]
        st.write(f"**{name}**: **{predicted_label}** (Confidence: {confidence:.4f})")

st.markdown("--- Say Hi to your Assistant! ---")
