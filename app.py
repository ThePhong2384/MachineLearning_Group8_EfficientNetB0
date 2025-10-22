import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import cv2

# 1. Load mô hình
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
    try:
        # Mở ảnh
        image = Image.open(uploaded_file)
        img = np.array(image)
        st.write("Shape ảnh gốc:", img.shape)

       
        # Luôn đảm bảo ảnh 3 kênh
        if img.ndim == 2:  # Grayscale HxW
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:  # Grayscale HxWx1
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        # Nếu RGB (3 kênh) thì không làm gì
        st.write("Shape ảnh sau khi chuyển về 3 kênh:", img.shape)


        # Resize về 224x224
        img_resized = cv2.resize(img, (224, 224))
        st.write("Sau khi resize: ", img_resized.shape)

        # Batch dimension
        img_array = np.expand_dims(img_resized, axis=0)
        st.write("Sau khi thêm batch dimension: ", img_array.shape)

        # Preprocess
        img_array = preprocess_input(img_array)
        if img_array.shape[-1] != 3:
            raise ValueError(f"Shape không đúng sau preprocess: {img_array.shape}. Mong đợi 3 kênh, nhưng nhận được {img_array.shape[-1]} kênh.")
        return img_array, img_resized  # Trả về array và ảnh hiển thị
    except Exception as e:
        st.error(f"Lỗi khi chuẩn bị ảnh: {str(e)}. Vui lòng kiểm tra ảnh upload.")
        return None, None

if uploaded_file is not None:
    # Hiển thị ảnh upload
    try:
        st.image(uploaded_file, caption="Ảnh gốc", use_column_width=True)
    except Exception as e:
        st.error(f"Lỗi khi hiển thị ảnh gốc: {str(e)}")

    # Chuẩn hóa ảnh
    img_array, img_resized = prepare_image(uploaded_file)

    if img_array is None or img_resized is None:
        st.stop()

    # Hiển thị ảnh chuẩn hóa (224x224x3)
    try:
        st.image(img_resized, caption="Ảnh sau khi xử lý (224x224)", use_column_width=True)
    except Exception as e:
        st.error(f"Lỗi khi hiển thị ảnh xử lý: {str(e)}")

    st.write("Shape ảnh sau xử lý:", img_array.shape)

    # --- 3. Dự đoán ---
    with st.spinner('Đang dự đoán...'):
        try:
            pred = model.predict(img_array)
            pred = tf.nn.softmax(pred).numpy()  # Chuyển thành probabilities
            if pred.shape[1] != 5:  # Kiểm tra số lớp
                raise ValueError(f"Số lớp dự đoán không khớp: {pred.shape[1]}. Mong đợi 5 lớp.")
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
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {str(e)}. Vui lòng kiểm tra mô hình hoặc ảnh.")
