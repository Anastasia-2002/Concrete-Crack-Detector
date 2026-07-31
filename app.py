import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
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
        'Dense': CustomDense  # Register our custom Dense layer for loading
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def check_input_shape(name, model):
    """Warn (don't crash) if a model's expected input resolution drifts from
    the IMAGE_HEIGHT/IMAGE_WIDTH constants used to preprocess uploads. This
    is the exact class of silent mismatch that caused the double-normalization
    bug this app previously shipped with -- fail loudly in the sidebar instead
    of silently producing a wrong-but-confident prediction."""
    try:
        shape = model.input_shape  # e.g. (None, 128, 128, 3)
        expected = (shape[1], shape[2])
    except Exception:
        return  # Shape not introspectable (e.g. exotic input spec) - skip check
    if expected != (IMAGE_HEIGHT, IMAGE_WIDTH):
        st.sidebar.warning(
            f"⚠️ {name} expects input {expected}, but the app resizes uploads "
            f"to {(IMAGE_HEIGHT, IMAGE_WIDTH)}. Predictions may be unreliable."
        )


# Model registry: add/remove entries here instead of duplicating load blocks.
MODEL_REGISTRY = [
    {"name": "Custom CNN", "path": "models/custom_cnn.h5"},
    {"name": "MobileNetV3 Transfer Learning", "path": "models/mobilenetv3_transfer.h5"},
]

loaded_models = {}

with st.spinner("Loading models..."):
    for entry in MODEL_REGISTRY:
        name, path = entry["name"], entry["path"]
        if not os.path.exists(path):
            st.sidebar.warning(f"{name} model not found at {path}")
            continue
        try:
            model = load_model(path)
            check_input_shape(name, model)
            loaded_models[name] = model
            st.sidebar.success(f"Loaded {name}")
        except Exception as e:
            st.sidebar.error(f"Error loading {name} model: {e}")


# --- Sidebar for model selection ---
st.sidebar.header("Model Selection")

model_to_use = None
selected_model_name = None

if loaded_models:
    selected_model_name = st.sidebar.radio(
        "Choose a model for prediction:",
        list(loaded_models.keys()),
        index=0  # Default to the first available model
    )
    model_to_use = loaded_models[selected_model_name]
else:
    st.error("No models were loaded. Please ensure model files are present in the 'models/' directory.")


# --- Image Upload and Prediction ---
st.header("Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_to_use is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
    except UnidentifiedImageError:
        st.error("That file doesn't look like a valid image. Please upload a JPG or PNG.")
        st.stop()
    except Exception as e:
        st.error(f"Couldn't read the uploaded file: {e}")
        st.stop()

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

    try:
        with st.spinner("Predicting..."):
            predictions = model_to_use.predict(img_array)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

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
    # Semantic, accessible result display (no injected HTML/CSS): red/warning
    # for a detected crack, green/success otherwise.
    if predicted_class_name == "Cracked":
        st.error(f"⚠️ The model predicts: **{predicted_class_name}**")
    else:
        st.success(f"✅ The model predicts: **{predicted_class_name}**")
    st.metric("Confidence", f"{confidence_pct:.2f}%")

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
