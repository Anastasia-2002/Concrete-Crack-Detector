import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Concrete Crack Detector",
    page_icon="🧱",
    layout="centered"
)

st.title("🧱 Concrete Crack Detector")
st.write("Upload an image of a concrete surface to detect whether it contains cracks.")

# -----------------------------------------------------------------------------
# 2. Path Resolution & Model Loading (Define BASE_DIR FIRST)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv3_transfer.keras")

@st.cache_resource
def load_crack_detector_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Please check that 'models/mobilenetv3_transfer.keras' "
            f"is uploaded to your GitHub repository."
        )
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)

try:
    model = load_crack_detector_model(MODEL_PATH)
except Exception as e:
    st.error("⚠️ Error loading the model.")
    st.error(f"Details: {e}")
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
