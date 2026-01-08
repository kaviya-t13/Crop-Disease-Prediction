import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from solutions import solutions

st.set_page_config(page_title="🌾 Crop Disease Detection", layout="centered")
st.markdown("""
    <h1 style="
        font-size: 50px;
        color: white;
        text-align: center;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.6);
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        border-radius: 10px;
        padding: 20px;
        ">
        🌾 Crop Disease Detection & Solution System
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
    <h3 style="text-align: center; color: #ddd;">
        Upload your leaf image to detect disease
    </h3>
""", unsafe_allow_html=True)


# 1️⃣ Load Model
model = tf.keras.models.load_model("crop_disease_mobilenet.h5")

# 2️⃣ Load Class Names
with open("class_names.json") as f:
    class_names = json.load(f)

# 3️⃣ Image Upload
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((224,224))
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # 4️⃣ Preprocess Image
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 5️⃣ Prediction
    prediction = model.predict(img_array)
    confidence = np.max(prediction)*100
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader(f"🦠 Detected Disease: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # 6️⃣ Show Solution
    
    if "healthy" in predicted_class.lower():
        st.success("🌱 Plant is Healthy")
    
    elif predicted_class in solutions:
        st.error("🦠 Disease Detected")
        st.write("**Cause:**", solutions[predicted_class]["cause"])
        st.write("**Solution:**", solutions[predicted_class]["solution"])
    
    else:
        st.warning("⚠️ Disease detected, but solution not available yet")
