import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time


st.write("Hello World")
# --- Global Configuration ---
IMAGE_SIZE = 128
# NOTE: This list MUST be in the exact order of your training labels!
# Assuming the order is: glioma (0), meningioma (1), notumor (2), pituitary (3)
UNIQUE_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary'] 

# --- Model Loading (Cached to run only once) ---
@st.cache_resource
def load_trained_model():
    """Loads the model and uses st.cache_resource to prevent reloading on every interaction."""
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model. Make sure 'model.h5' is in the same directory. Error: {e}")
        return None

model = load_trained_model()

# --- Preprocessing Function (Mimics Training Pipeline) ---
def preprocess_image(image):
    """Resizes, converts to array, normalizes, and adds batch dimension."""
    # 1. Resize
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # 2. Convert to Array (NumPy)
    img_array = np.array(image, dtype=np.float32)
    
    # 3. Normalize (Crucial: must match training's 0-1 range)
    img_array = img_array / 255.0
    
    # 4. Expand Dimensions (Add the batch dimension)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- Main Streamlit Application ---
st.title("🧠 Brain Tumor Detection System")
st.markdown("Upload an MRI scan to get a real-time classification using the VGG16 Transfer Learning model.")

if model is not None:
    uploaded_file = st.file_uploader("Choose an MRI image (.jpg or .png)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded MRI Scan', use_column_width=True)

        st.subheader("Analysis Results:")
        
        # Preprocess and predict
        processed_img = preprocess_image(image)
        
        # Run prediction
        with st.spinner('Analyzing scan...'):
            time.sleep(1) # Simulate processing time
            predictions = model.predict(processed_img)

        # Extract the highest prediction and confidence
        predicted_class_index = np.argmax(predictions[0])
        confidence_score = predictions[0][predicted_class_index]
        predicted_label = UNIQUE_LABELS[predicted_class_index]

        # --- Display Final Result ---
        if predicted_label == 'notumor':
            st.success("✅ Prediction: No Tumor Detected")
            st.write(f"Confidence: **{confidence_score * 100:.2f}%**")
        else:
            st.error(f"🚨 Prediction: **{predicted_label.upper()}** Detected")
            st.write(f"Confidence in diagnosis: **{confidence_score * 100:.2f}%**")
            st.caption(f"Note: The model classified this image as index {predicted_class_index}.")
else:
    st.error("Cannot run prediction. Please ensure 'model.h5' is correctly loaded.")