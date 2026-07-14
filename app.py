import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

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

# Türkçe isim + açıklama + öneri
disease_info = {
    "Tomato_Bacterial_spot": {
        "tr_name": "Bakteriyel Leke Hastalığı",
        "description": "Yapraklarda küçük, koyu kahverengi/siyah lekeler görülür. Nemli ve sıcak havalarda hızla yayılır.",
        "suggestion": "Bakır bazlı fungisit uygulayın, sulama sırasında yaprakları ıslatmaktan kaçının, hastalıklı yaprakları uzaklaştırın."
    },
    "Tomato_Early_blight": {
        "tr_name": "Erken Yanıklık (Alternaria)",
        "description": "Yapraklarda konsantrik halkalı kahverengi lekeler oluşur, alt yapraklardan başlar.",
        "suggestion": "Bitki aralarını iyi havalandırın, uygun fungisit kullanın, düşük azotlu gübreleme yapın."
    },
    "Tomato_Late_blight": {
        "tr_name": "Geç Yanıklık (Phytophthora)",
        "description": "Yapraklarda büyük, düzensiz, su emmiş görünümlü koyu lekeler oluşur. Çok hızlı yayılabilir ve ciddi kayıplara yol açabilir.",
        "suggestion": "Acilen etkilenen bitkileri izole edin, uygun fungisitle müdahale edin, nemli ortamlardan kaçının."
    },
    "Tomato_Leaf_Mold": {
        "tr_name": "Yaprak Küfü",
        "description": "Yaprak üstünde soluk sarı lekeler, altında ise kadifemsi zeytin-yeşili küf tabakası oluşur.",
        "suggestion": "Sera/örtü altı yetiştiricilikte nem oranını düşürün, havalandırmayı artırın, fungisit uygulayın."
    },
    "Tomato_Septoria_leaf_spot": {
        "tr_name": "Septoria Yaprak Lekesi",
        "description": "Küçük, yuvarlak, gri merkezli ve koyu kenarlı lekeler yaprağı kaplar.",
        "suggestion": "Hastalıklı yaprakları toplayıp uzaklaştırın, fungisit uygulayın, sulamada yaprakları ıslatmayın."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "tr_name": "Kırmızı Örümcek (İki Noktalı)",
        "description": "Yapraklarda sararma, noktalı desenler ve ince örümcek ağları görülür. Kuru ve sıcak havada artış gösterir.",
        "suggestion": "Akarisit uygulayın, bitkiyi düzenli su ile yıkayın/nemlendirin, doğal düşmanlardan (uğur böceği vb.) faydalanın."
    },
    "Tomato_Target_Spot": {
        "tr_name": "Hedef Lekesi",
        "description": "Konsantrik halkalı, hedef tahtası görünümünde kahverengi lekeler oluşur.",
        "suggestion": "Uygun fungisit uygulayın, bitki artıklarını temizleyin, ekim nöbeti uygulayın."
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "tr_name": "Sarı Yaprak Kıvırcıklığı Virüsü",
        "description": "Yapraklar sararır, küçülür ve kıvrılır. Beyazsinek tarafından taşınan viral bir hastalıktır.",
        "suggestion": "Beyazsinek popülasyonunu kontrol altına alın, hastalıklı bitkileri sökün, dayanıklı çeşitler tercih edin."
    },
    "Tomato_Tomato_mosaic_virus": {
        "tr_name": "Mozaik Virüsü",
        "description": "Yapraklarda açık-koyu yeşil mozaik desenli renk bozukluğu ve şekil bozulması görülür.",
        "suggestion": "Hastalıklı bitkileri imha edin, alet-ekipmanı dezenfekte edin, sağlıklı tohum/fide kullanın."
    },
    "Tomato_healthy": {
        "tr_name": "Sağlıklı Yaprak",
        "description": "Yaprakta herhangi bir hastalık belirtisi tespit edilmedi.",
        "suggestion": "Düzenli bakım ve gözlemlere devam edin. Bitkiniz sağlıklı görünüyor! 🌱"
    }
}

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
    info = disease_info[predicted_class]

    st.markdown("---")

    if predicted_class == "Tomato_healthy":
        st.success(f"✅ **{info['tr_name']}**")
    else:
        st.error(f"⚠️ **{info['tr_name']}**")

    st.write(f"**Güven skoru:** {confidence:.2f}%")
    st.write(f"**Açıklama:** {info['description']}")
    st.write(f"**Öneri:** {info['suggestion']}")

    st.markdown("---")
    st.subheader("📊 Tüm sınıf olasılıkları")

    # Bar chart için veri hazırla (Türkçe isimlerle, yüzde olarak, azalan sırada)
    chart_data = pd.DataFrame({
        "Sınıf": [disease_info[c]["tr_name"] for c in classes],
        "Olasılık (%)": [float(p) * 100 for p in prediction[0]]
    }).sort_values("Olasılık (%)", ascending=False).set_index("Sınıf")

    st.bar_chart(chart_data)

    with st.expander("Ham veriyi gör"):
        for cls, prob in sorted(zip(classes, prediction[0]), key=lambda x: -x[1]):
            st.write(f"{disease_info[cls]['tr_name']}: {prob*100:.2f}%")
