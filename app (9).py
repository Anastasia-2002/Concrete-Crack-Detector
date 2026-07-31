import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd # Added for DataFrame display

# Define constants (should match your training setup)
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CLASS_NAMES = ['Decks', 'Walls'] # Adjust if your class names are different or loaded dynamically

# --- Page Configuration ---
st.set_page_config(
    page_title="Crack Detection App",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🔍 Crack Detection Application")
st.write("Upload an image and let the models predict if it's cracked or non cracked.")

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

# Paths to your saved models (assuming they are in the same directory as app.py or accessible)
CNN_MODEL_PATH = 'models/custom_cnn.h5'
TL_MODEL_PATH = 'models/mobilenetv3_transfer.h5'

custom_cnn_model = None
transfer_learning_model = None

with st.spinner("Loading models..."):
    if os.path.exists(CNN_MODEL_PATH):
        try:
            custom_cnn_model = load_model(CNN_MODEL_PATH)
            st.sidebar.success(f"Loaded Custom CNN from {CNN_MODEL_PATH}")
        except Exception as e:
            st.sidebar.error(f"Error loading Custom CNN model: {e}")
    else:
        st.sidebar.warning(f"Custom CNN model not found at {CNN_MODEL_PATH}")
    if os.path.exists(TL_MODEL_PATH):
        try:
            transfer_learning_model = load_model(TL_MODEL_PATH)
            st.sidebar.success(f"Loaded MobileNetV3 Transfer Learning model from {TL_MODEL_PATH}")
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
    st.error("No models were loaded. Please ensure model files are present.")


# --- Image Upload and Prediction ---
st.header("Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_to_use is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess image for prediction
    img_array = np.array(image.resize((IMAGE_WIDTH, IMAGE_HEIGHT)))
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    with st.spinner("Predicting..."):
        predictions = model_to_use.predict(img_array)

    # Determine if it's binary (sigmoid) or multiclass (softmax)
    if model_to_use.output_shape[-1] == 1: # Assuming binary classification with a single output neuron
        score = predictions[0][0]
        predicted_class_index = 1 if score >= 0.5 else 0 # Assuming 1 is the positive class
        confidence = score if predicted_class_index == 1 else (1 - score)
    else: # Multiclass classification with softmax output (e.g., num_classes > 1 in output layer)
        predicted_class_index = np.argmax(predictions)
        confidence = np.max(predictions)

    predicted_class_name = CLASS_NAMES[predicted_class_index]

    st.subheader("Prediction Results")
    st.markdown(f"The model predicts: **<span style='color:blue;'>{predicted_class_name}</span>** with a confidence of **{confidence:.2f}%**", unsafe_allow_html=True)

    # Display raw prediction scores (optional)
    st.write("Raw prediction scores:")
    pred_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": predictions[0]})
    st.dataframe(pred_df.style.format({'Probability': '{:.4f}'}))


# --- Instructions to run the app ---
st.sidebar.markdown("""
---
### How to Run this App:
1.  **Ensure you have `custom_cnn.h5` and `mobilenetv3_transfer.h5` in a folder named `models/` in the same directory as this `app.py` in your GitHub repository.**
2.  Deploy your GitHub repository containing this `app.py` and the `models/` folder to Streamlit.
""")
