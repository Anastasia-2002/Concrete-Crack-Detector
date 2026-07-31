import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

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
MODEL_PATH = os.path.join(BASE_DIR, "models", "custom_cnn.keras")

@st.cache_resource
def load_crack_detector_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Please check that 'models/custom_cnn.keras' "
            f"is uploaded to your GitHub repository."
        )
    return tf.keras.models.load_model(path, safe_mode=False)

try:
    model = load_crack_detector_model(MODEL_PATH)
except Exception as e:
    st.error("⚠️ Error loading the model.")
    st.error(f"Details: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Image Preprocessing & Prediction Helper
# -----------------------------------------------------------------------------
def predict_image(image, model_instance):
    # Determine model input shape dynamically (defaults to 228x228 if unspecified)
    input_shape = model_instance.input_shape
    target_height = input_shape[1] if input_shape[1] is not None else 228
    target_width = input_shape[2] if input_shape[2] is not None else 228

    # Auto-orient image (fixes EXIF rotation issues from phone cameras)
    image = ImageOps.exif_transpose(image)

    # Convert image to standard 3-channel RGB (handles RGBA, grayscale, indexed PNG, BMP, WEBP, etc.)
    img = image.convert("RGB")
    img = img.resize((target_width, target_height))
    
    # Convert image to numpy array and normalize pixel values (0 to 1)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Run inference
    prediction = model_instance.predict(img_array)
    return prediction

# -----------------------------------------------------------------------------
# 4. User Interface & File Upload (Accepts Any File Type)
# -----------------------------------------------------------------------------
# Omitting 'type' parameter allows the uploader to accept any file uploaded by the user
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=None
)

if uploaded_file is not None:
    try:
        # Attempt to open as an image
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Concrete Surface"):
            with st.spinner("Analyzing image for structural cracks..."):
                raw_pred = predict_image(image, model)
                
                # Extract score depending on output shape (sigmoid binary vs 2-class softmax)
                if raw_pred.shape[-1] == 1:
                    confidence = float(raw_pred[0][0])
                    is_crack = confidence > 0.5
                else:
                    confidence = float(np.max(raw_pred))
                    is_crack = int(np.argmax(raw_pred)) == 1

                # Display Results
                st.markdown("---")
                if is_crack:
                    st.error("🚨 **Crack Detected!**")
                else:
                    st.success("✅ **No Crack Detected.**")

    except Exception as err:
        st.error("⚠️ Invalid file format. Please upload a valid image file.")
        st.exception(err)
