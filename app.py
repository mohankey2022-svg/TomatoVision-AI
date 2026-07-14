import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="TomatoVision AI",
    page_icon="🍅",
    layout="centered"
)
st.title("🍅 TomatoVision AI")
st.write("Deep Learning ile Domates Yaprağı Hastalık Tespiti")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("TomatoVisionAI_EfficientNetB0.keras")

model = load_model()

classes = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

uploaded_file = st.file_uploader(
    "🍅 Domates yaprağı fotoğrafı yükleyin",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen görüntü", use_container_width=True)

    img_size = (150, 150)
    img_resized = image.resize(img_size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analiz ediliyor..."):
        prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction[0])
    predicted_class = classes[predicted_index]
    confidence = float(np.max(prediction[0])) * 100

    st.success(f"**Tahmin:** {predicted_class}")
    st.write(f"**Güven skoru:** {confidence:.2f}%")

    st.subheader("Tüm sınıf olasılıkları")
    for cls, prob in sorted(zip(classes, prediction[0]), key=lambda x: -x[1]):
        st.write(f"{cls}: {prob*100:.2f}%")
