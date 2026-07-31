import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

# Define constants (should match your training setup)
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CLASS_NAMES = ['Cracked', 'Non-cracked']

# --- Page Configuration ---
st.set_page_config(
    page_title="Crack Detection App",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🔍 Crack Detection Application")
st.write("Upload an image and let the models predict if it's Cracked or Non-cracked.")

# Custom object to handle potential 'quantization_config' issue during model loading
class CustomDense(tf.keras.layers.Dense):
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        # Filter out unrecognized kwargs like 'quantization_config'
        kwargs.pop('quantization_config', None)
        super().__init__(units, activation=activation, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)

# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    """Loads a TensorFlow Keras model, handling potential custom objects."""
    custom_objects = {
        'Dense': CustomDense # Register our custom Dense layer for loading
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Paths to your saved models (relative to app.py in your GitHub repo)
CNN_MODEL_PATH = 'models/custom_cnn.h5'
TL_MODEL_PATH = 'models/mobilenetv3_transfer.h5'

custom_cnn_model = None
transfer_learning_model = None

with st.spinner("Loading models..."):
    if os.path.exists(CNN_MODEL_PATH):
        try:
            custom_cnn_model = load_model(CNN_MODEL_PATH)
            st.sidebar.success("Loaded Custom CNN")
        except Exception as e:
            st.sidebar.error(f"Error loading Custom CNN model: {e}")
    else:
        st.sidebar.warning(f"Custom CNN model not found at {CNN_MODEL_PATH}")

    if os.path.exists(TL_MODEL_PATH):
        try:
            transfer_learning_model = load_model(TL_MODEL_PATH)
            st.sidebar.success("Loaded MobileNetV3 Transfer Learning model")
        except Exception as e:
            st.sidebar.error(f"Error loading MobileNetV3 TL model: {e}")
    else:
        st.sidebar.warning(f"MobileNetV3 TL model not found at {TL_MODEL_PATH}")


# --- Sidebar for model selection ---
st.sidebar.header("Model Selection")

model_options = []
if custom_cnn_model: model_options.append('Custom CNN')
if transfer_learning_model: model_options.append('MobileNetV3 Transfer Learning')

model_to_use = None
selected_model_name = None

if model_options:
    selected_model_name = st.sidebar.radio(
        "Choose a model for prediction:",
        model_options,
        index=0 # Default to the first available model
    )

    if selected_model_name == 'Custom CNN':
        model_to_use = custom_cnn_model
    elif selected_model_name == 'MobileNetV3 Transfer Learning':
        model_to_use = transfer_learning_model
else:
    st.error("No models were loaded. Please ensure model files are present in the 'models/' directory.")


# --- Image Upload and Prediction ---
st.header("Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_to_use is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")

    # Preprocess image for prediction: resize only. Both saved models already
    # contain their own Rescaling layer as the first layer (custom_cnn.h5 does
    # x/255, mobilenetv3_transfer.h5 does x/127.5 - 1), so the input here must
    # stay in raw [0, 255] range. Dividing by 255 here as well as inside the
    # model double-normalizes the image and silently produces near-constant,
    # meaningless activations -> wrong predictions with high confidence.
    img_array = np.array(image.resize((IMAGE_WIDTH, IMAGE_HEIGHT)), dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    with st.spinner("Predicting..."):
        predictions = model_to_use.predict(img_array)

    # Determine if output is single sigmoid neuron or 2-class softmax
    if model_to_use.output_shape[-1] == 1:
        score = float(predictions[0][0])
        predicted_class_index = 1 if score >= 0.5 else 0
        confidence = score if predicted_class_index == 1 else (1.0 - score)
        probs = [1.0 - score, score]
    else:
        probs = predictions[0]
        predicted_class_index = int(np.argmax(probs))
        confidence = float(probs[predicted_class_index])

    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence_pct = confidence * 100.0

    st.subheader("Prediction Results")
    st.markdown(
        f"The model predicts: **<span style='color:blue;'>{predicted_class_name}</span>** "
        f"with a confidence of **{confidence_pct:.2f}%**", 
        unsafe_allow_html=True
    )

    # Display raw prediction scores
    st.write("Raw prediction scores:")
    pred_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probs})
    st.dataframe(pred_df.style.format({'Probability': '{:.4f}'}))


# --- Sidebar Instructions ---
st.sidebar.markdown("""
---
### How to Run this App:
1. Ensure your GitHub repository contains:
    * `app.py` (this file)
    * A `models/` directory containing `custom_cnn.h5` and `mobilenetv3_transfer.h5`
    * `requirements.txt` with `streamlit`, `tensorflow`, `numpy`, `Pillow`, and `pandas`
2. Deploy your GitHub repository to Streamlit Cloud.
""")
