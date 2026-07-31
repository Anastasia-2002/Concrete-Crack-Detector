import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration Settings
# -----------------------------------------------------------------------------
IMAGE_WIDTH = 224   # Adjust if your model was trained on a different size
IMAGE_HEIGHT = 224
CLASS_NAMES = ["Negative", "Positive"]  # Adjust labels if needed (e.g., ['No Crack', 'Crack'])

# -----------------------------------------------------------------------------
# 1. Page Configuration & Title
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Concrete Crack Detector",
    page_icon="🧱",
    layout="centered"
)

st.title("🧱 Concrete Crack Detector")
st.write("Upload an image of a concrete surface to detect whether it contains cracks.")

# -----------------------------------------------------------------------------
# 2. Path Resolution & Model Loading
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv3_transfer.keras")

@st.cache_resource
def load_crack_detector_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Please ensure 'models/mobilenetv3_transfer.keras' "
            f"is uploaded to your GitHub repository."
        )
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)

try:
    model = load_crack_detector_model(MODEL_PATH)
except Exception as e:
    st.error("⚠️ Error loading the model.")
    st.error(f"Details: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. File Uploader & Prediction Logic
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose an image for classification...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess image
    img_resized = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    st.subheader("Prediction Results:")

    try:
        predictions = model.predict(img_array, verbose=0)
        
        # Handle binary (1 output node) vs multi-class (2+ output nodes)
        if predictions.shape[-1] == 1:
            score = float(predictions[0][0])
            if score > 0.5:
                predicted_label = CLASS_NAMES[1] if len(CLASS_NAMES) > 1 else "Crack"
                confidence = score
            else:
                predicted_label = CLASS_NAMES[0] if len(CLASS_NAMES) > 0 else "No Crack"
                confidence = 1.0 - score
        else:
            preds = predictions[0]
            predicted_class_index = np.argmax(preds)
            predicted_label = CLASS_NAMES[predicted_class_index]
            confidence = float(preds[predicted_class_index])

        st.write(f"**MobileNetV3 TL**: **{predicted_label}** (Confidence: {confidence:.4f})")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("Concrete Crack Detector Assistant")
