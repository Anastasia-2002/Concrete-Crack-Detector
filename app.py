import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Streamlit App Configuration --- #
st.set_page_config(page_title="Image Classification App", layout="centered")

st.title("Image Classification: Cracked vs. Non-cracked Decks")
st.write("Upload an image to classify it as 'Cracked' or 'Non-cracked' using two trained models: a Custom CNN and a MobileNetV3 Transfer Learning model.")

# --- Global Parameters (must match training parameters) --- #
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
class_names = ['Cracked', 'Non-cracked'] # Ensure this order matches your model's output

# Path to saved models (ensure these paths are correct relative to where app.py is run)
# Assuming models are in a 'models' directory in the same location as app.py
CNN_MODEL_PATH = 'models/custom_cnn.keras'
TL_MODEL_PATH = 'models/mobilenetv3_transfer.keras'

# --- Model Loading with Caching --- #
@st.cache_resource
def load_models():
    """Loads the trained Keras models with Streamlit's caching."""
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
        st.error(f"Error loading models: {e}. Please ensure 'models/custom_cnn.keras' and 'models/mobilenetv3_transfer.keras' exist.")
        return None, None

cnn_model, tl_model = load_models()

# Stop the app if models failed to load
if cnn_model is None or tl_model is None:
    st.stop()

# --- Image Upload and Prediction --- #
uploaded_file = st.file_uploader("Choose an image for classification...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    # Resize image to target dimensions
    img_resized = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    # Convert to NumPy array
    img_array = tf.keras.utils.img_to_array(img_resized)  # [0, 255] float32
    # Add batch dimension (1, H, W, 3)
    img_array = np.expand_dims(img_array, axis=0)

    st.subheader("Prediction Results:")

    models_to_predict = {'Custom CNN': cnn_model, 'MobileNetV3 TL': tl_model}

    for name, model in models_to_predict.items():
        # The models handle their own preprocessing (rescaling, mobilenet_v3.preprocess_input)
        # as built during training.
        predictions = model.predict(img_array, verbose=0)[0] # Get probabilities for all classes
        # For binary classification, predictions is often a single value (prob of class 1)
        # For multi-class, it's an array of probabilities. Here, assuming binary output (2 classes)
        # where models.predict gives a probability for each class, e.g., [prob_class0, prob_class1]
        
        # Adjusting based on `predict_image` function which used [0][0] and assumed binary with 0.5 threshold
        # The current Streamlit app code is already set up for multi-class where `predictions` is an array of probs.
        predicted_class_index = np.argmax(predictions) # Get the index of the class with the highest probability
        predicted_label = class_names[predicted_class_index]
        confidence = predictions[predicted_class_index] # Confidence of the predicted class

        st.write(f"**{name}**: **{predicted_label}** (Confidence: {confidence:.4f})")

st.markdown("--- Say Hi to your Assistant! ---")

# Save app.py to Google Drive
output_dir = '/content/drive/MyDrive/streamlit_app'
os.makedirs(output_dir, exist_ok=True)
app_file_path = os.path.join(output_dir, 'app.py')

# Get the content of the current cell and save it to app.py
# This assumes the Streamlit app code is contained within this single cell
# For this to work, the magic function %%writefile is usually used at the start of the cell
# However, since we are programmatically writing, we need to get the code from the cell itself
# Since I don't have direct access to `%%writefile` for the *current* cell programmatically
# I will hardcode the content to be saved as the app.py string.
app_content = """
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Image Classification App", layout="centered")
st.title("Image Classification: Cracked vs. Non-cracked Decks")
st.write("Upload an image to classify it as 'Cracked' or 'Non-cracked' using two trained models.")

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
class_names = ['Cracked', 'Non-cracked']
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
"""

with open(app_file_path, 'w') as f:
    f.write(app_content)
print(f"Streamlit app.py saved to: {app_file_path}")

# Generate requirements.txt
requirements_file_path = os.path.join(output_dir, 'requirements.txt')
# Using pip freeze to get exact versions, and then filtering for common ML/Streamlit packages
# For simplicity, will just include the main dependencies.
# In a real scenario, you'd run !pip freeze > requirements.txt and filter/edit.
requirements_content = """
streamlit
tensorflow
numpy
Pillow
"""
with open(requirements_file_path, 'w') as f:
    f.write(requirements_content)
print(f"requirements.txt saved to: {requirements_file_path}")

print("\nTo run your Streamlit app from Colab, you can use:")
print(f"!streamlit run {app_file_path} & npx localtunnel --port 8501")
print("Or, if you prefer to run it locally, download `app.py` and `requirements.txt` and run `streamlit run app.py` in your terminal.")
