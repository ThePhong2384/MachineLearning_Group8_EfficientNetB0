import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# 1. Load mô hình đã huấn luyện
MODEL_PATH = "best_model.keras"

@st.cache_resource
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

model = load_trained_model(MODEL_PATH)

class_names = ["cat", "cattle", "chicken", "dog", "elephant"]

# 2. Hàm tiền xử lý ảnh 
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 3. Giao diện Streamlit 
st.title("🐾 PHÂN LOẠI HÌNH ẢNH ĐỘNG VẬT 🐾")
st.write("Tải lên một bức ảnh để mô hình dự đoán lớp động vật.")

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh tải lên", use_container_width=True)

    img_array = preprocess_image(image)

    # Dự đoán
    preds = model.predict(img_array)
    pred_class_idx = np.argmax(preds, axis=1)[0]
    pred_class_name = class_names[pred_class_idx]
    pred_confidence = preds[0][pred_class_idx]

    # --- Ngưỡng xác suất ---
    CONFIDENCE_THRESHOLD = 0.6  # điều chỉnh tùy mô hình

    if pred_confidence < CONFIDENCE_THRESHOLD:
        st.error("❌ Ảnh không thuộc 5 lớp: cat, cattle, chicken, dog, elephant. Vui lòng tải ảnh khác")
    else:
        st.success(f"Dự đoán: **{pred_class_name}** với độ chính xác {pred_confidence*100:.2f}%")
