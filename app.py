import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import cv2

#  1. Load mô hình
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('best_model.keras')
        st.success("Đã tải mô hình thành công!")
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {str(e)}. Vui lòng kiểm tra file 'best_model.keras'.")
        return None

model = load_model()
if model is None:
    st.stop()

# 2. Streamlit giao diện
st.title("Ứng Dụng Phân Loại Động Vật với EfficientNetB0")
st.markdown("Upload hình ảnh động vật (jpg, jpeg, png) để dự đoán. Mô hình hỗ trợ 5 lớp: mèo, bò, gà, chó, voi.")

uploaded_file = st.file_uploader("Chọn ảnh động vật...", type=["jpg", "jpeg", "png"])

def prepare_image(uploaded_file):
    # Mở ảnh
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Convert mọi ảnh về RGB 3 kênh
    if len(img.shape) == 2:  # Grayscale HxW
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:  # Grayscale HxWx1
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        alpha_channel = img[:, :, 3]
        rgb_channels = img[:, :, :3]
        white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32)/255.0
        alpha_factor = np.concatenate([alpha_factor]*3, axis=2)
        img = (rgb_channels.astype(np.float32)*alpha_factor + white_background*(1-alpha_factor)).astype(np.uint8)

    # Resize về 224x224
    img_resized = cv2.resize(img, (224,224))
    # Batch dimension
    img_array = np.expand_dims(img_resized, axis=0)
    # Preprocess
    img_array = preprocess_input(img_array)
    return img_array, img_resized  # Trả về array và ảnh hiển thị

if uploaded_file is not None:
    # Hiển thị ảnh upload
    st.image(uploaded_file, caption="Ảnh gốc", use_column_width=True)

    # Chuẩn hóa ảnh
    img_array, img_resized = prepare_image(uploaded_file)

    # Hiển thị ảnh chuẩn hóa (224x224x3)
    st.image(img_resized, caption="Ảnh sau khi xử lý (224x224)", use_column_width=True)

    st.write("Shape ảnh sau xử lý:", img_array.shape)

    # --- 3. Dự đoán ---
    with st.spinner('Đang dự đoán...'):
        pred = model.predict(img_array)
        pred = tf.nn.softmax(pred).numpy()  # Chuyển thành probabilities
        class_idx = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred)

        # Class names (khớp với LabelEncoder từ notebook)
        class_names = ['cat', 'cattle', 'chicken', 'dog', 'elephant']
        predicted_label = class_names[class_idx]

        # --- 4. Hiển thị kết quả ---
        st.success(f"Dự đoán: {predicted_label} ({confidence*100:.2f}% confidence)")

        # Hiển thị top-3 dự đoán
        st.subheader("Top 3 dự đoán:")
        top_3_idx = np.argsort(pred[0])[-3:][::-1]
        for idx in top_3_idx:
            st.write(f"{class_names[idx]}: {pred[0][idx]*100:.2f}%")