import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Page Configuration & Title
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Concrete Crack Detector", page_icon="🧱", layout="centered")

st.title("🧱 Concrete Crack Detector")
st.write("Upload an image of a concrete surface to detect whether it contains cracks.")

# -----------------------------------------------------------------------------
# 2. Parameters & Model Path
# -----------------------------------------------------------------------------
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CLASS_NAMES = ['Decks', 'Walls']  # Adjust if your classes are ['No Crack', 'Crack']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TL_MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv3_transfer.keras")

# -----------------------------------------------------------------------------
# 3. Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(TL_MODEL_PATH):
        st.error(f"❌ Missing model file at: `{TL_MODEL_PATH}`")
        st.stop()
    try:
        model = tf.keras.models.load_model(TL_MODEL_PATH, compile=False, safe_mode=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# -----------------------------------------------------------------------------
# 4. Inference Logic
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose an image for classification...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess image
    img_resized = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    st.subheader("Prediction Results:")

    try:
        predictions = model.predict(img_array, verbose=0)
        
        if predictions.shape[-1] == 1:
            score = float(predictions[0][0])
            idx = 1 if score > 0.5 else 0
            confidence = score if score > 0.5 else 1.0 - score
        else:
            preds = predictions[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx])

        predicted_label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
        st.write(f"**MobileNetV3 TL Prediction**: **{predicted_label}** (Confidence: {confidence:.4f})")
        
    except Exception as e:
        st.error(f"Error during evaluation: {e}")

st.markdown("---")
st.caption("Concrete Crack Detector Assistant")
