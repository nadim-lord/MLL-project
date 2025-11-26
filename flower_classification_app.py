import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# --------------------------
# Load Model & Class Labels
# --------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

@st.cache_resource
def load_class_indices():
    with open("class_indices.json", "r") as f:
        return json.load(f)

model = load_model()
class_indices = load_class_indices()
classes = list(class_indices.keys())

# --------------------------
# Streamlit UI
# --------------------------
st.title("🌸 Flower Classification App")
st.write("Upload an image and the model will predict the flower category.")

uploaded_file = st.file_uploader("Upload flower image", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_img = preprocess_image(img)

    # Predict
    prediction = model.predict(input_img)[0]
    pred_idx = np.argmax(prediction)
    pred_class = classes[pred_idx]
    confidence = prediction[pred_idx] * 100

    st.subheader("Prediction Result")
    st.write(f"**Flower Type:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
