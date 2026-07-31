import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Page Configuration & Title
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Image Classification App", layout="centered")

st.title("🧱 Concrete Crack Detector")
st.write("Upload an image to classify it using the Custom CNN and MobileNetV3 models.")

# -----------------------------------------------------------------------------
# 2. Parameters & Relative Path Setup
# -----------------------------------------------------------------------------
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CLASS_NAMES = ['Decks', 'Walls']  # Adjust order/labels if needed

# Dynamic path resolution (Works on Streamlit Cloud, Windows, Mac, and Linux)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "custom_cnn.keras")
TL_MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv3_transfer.keras")

# -----------------------------------------------------------------------------
# 3. Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Loads the trained Keras models with Streamlit caching."""
    with st.spinner("Loading models..."):
        if not os.path.exists(CNN_MODEL_PATH):
            st.error(f"❌ Missing file: `{CNN_MODEL_PATH}`")
            return None, None
        if not os.path.exists(TL_MODEL_PATH):
            st.error(f"❌ Missing file: `{TL_MODEL_PATH}`")
            return None, None

        try:
            cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False, safe_mode=False)
            tl_model = tf.keras.models.load_model(TL_MODEL_PATH, compile=False, safe_mode=False)
            st.success("Models loaded successfully!")
            return cnn_model, tl_model
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None

cnn_model, tl_model = load_models()

# Stop app execution if loading fails
if cnn_model is None or tl_model is None:
    st.stop()

# -----------------------------------------------------------------------------
# 4. Inference & UI Logic
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose an image for classification...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocessing
    img_resized = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    st.subheader("Prediction Results:")

    models_to_predict = {'Custom CNN': cnn_model, 'MobileNetV3 TL': tl_model}

    for name, model in models_to_predict.items():
        try:
            predictions = model.predict(img_array, verbose=0)
            
            # Binary output (1 probability node)
            if predictions.shape[-1] == 1:
                score = float(predictions[0][0])
                idx = 1 if score > 0.5 else 0
                confidence = score if score > 0.5 else 1.0 - score
            # Multi-class output (2+ probability nodes)
            else:
                preds = predictions[0]
                idx = int(np.argmax(preds))
                confidence = float(preds[idx])

            predicted_label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
            st.write(f"**{name}**: **{predicted_label}** (Confidence: {confidence:.4f})")
            
        except Exception as e:
            st.error(f"Error evaluating {name}: {e}")

st.markdown("---")
st.caption("Concrete Crack Detector Assistant")
