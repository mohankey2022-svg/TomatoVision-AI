import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(
    page_title="TomatoVision AI",
    page_icon="🍅",
    layout="centered"
)

# ---------- Tasarım: özel CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Work+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Work Sans', sans-serif;
}

.stApp {
    background-color: #F6F2E9;
    color: #202B22;
}

#MainMenu, footer, header {visibility: hidden;}

.tv-banner {
    background: #1F3B2C;
    margin: -1rem -1rem 1.75rem -1rem;
    padding: 2.25rem 1.5rem 1.75rem 1.5rem;
    border-bottom: 4px solid #C1432E;
}
.tv-banner h1 {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.4rem;
    color: #F6F2E9;
    margin: 0;
    letter-spacing: -0.01em;
}
.tv-banner p {
    font-family: 'Work Sans', sans-serif;
    color: #B9C9BC;
    margin: 0.35rem 0 0 0;
    font-size: 1.02rem;
}

[data-testid="stFileUploader"] {
    background: #EFE8D8;
    border: 2px dashed #4F9D5C;
    border-radius: 14px;
    padding: 1rem;
}

.tv-card {
    background: #EFE8D8;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-top: 1rem;
    border-left: 6px solid #4F9D5C;
}
.tv-card.disease {
    border-left: 6px solid #C1432E;
}
.tv-card h2 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.5rem;
    margin: 0 0 0.6rem 0;
    color: #202B22;
}
.tv-card .tv-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #5B6B5D;
    margin-bottom: 0.15rem;
}
.tv-card p {
    font-size: 0.97rem;
    line-height: 1.5;
    margin: 0.2rem 0 0.9rem 0;
}

.tv-section-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.25rem;
    margin: 1.75rem 0 0.5rem 0;
    color: #1F3B2C;
}

.tv-footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #8A968B;
    text-align: center;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #DDD5C2;
}
</style>
""", unsafe_allow_html=True)

# ---------- Banner ----------
st.markdown("""
<div class="tv-banner">
    <h1>🍅 max01</h1>
    <p>Domates Yaprağı Hastalık Tespiti</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("TomatoVisionAI_EfficientNetB0.keras")

model = load_model()

classes = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

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
        "description": "Yapraklarda büyük, düzensiz, su emmiş görünümlü koyu lekeler oluşur. Çok hızlı yayılabilir.",
        "suggestion": "Etkilenen bitkileri izole edin, uygun fungisitle müdahale edin, nemli ortamlardan kaçının."
    },
    "Tomato_Leaf_Mold": {
        "tr_name": "Yaprak Küfü",
        "description": "Yaprak üstünde soluk sarı lekeler, altında kadifemsi zeytin-yeşili küf tabakası oluşur.",
        "suggestion": "Nem oranını düşürün, havalandırmayı artırın, fungisit uygulayın."
    },
    "Tomato_Septoria_leaf_spot": {
        "tr_name": "Septoria Yaprak Lekesi",
        "description": "Küçük, yuvarlak, gri merkezli ve koyu kenarlı lekeler yaprağı kaplar.",
        "suggestion": "Hastalıklı yaprakları toplayıp uzaklaştırın, fungisit uygulayın, sulamada yaprakları ıslatmayın."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "tr_name": "Kırmızı Örümcek (İki Noktalı)",
        "description": "Yapraklarda sararma, noktalı desenler ve ince örümcek ağları görülür.",
        "suggestion": "Akarisit uygulayın, bitkiyi düzenli nemlendirin, doğal düşmanlardan faydalanın."
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

uploaded_file = st.file_uploader("🍅 Domates yaprağı fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Yüklenen görüntü", use_container_width=True)

    img_resized = image.resize((150, 150))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    with st.spinner("Analiz ediliyor..."):
        prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction[0])
    predicted_class = classes[predicted_index]
    confidence = float(np.max(prediction[0])) * 100
    info = disease_info[predicted_class]
    is_healthy = predicted_class == "Tomato_healthy"
    accent = "#4F9D5C" if is_healthy else "#C1432E"

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"family": "IBM Plex Mono", "size": 30, "color": "#202B22"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8A968B"},
                "bar": {"color": accent},
                "bgcolor": "#EFE8D8",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#E4DCC8"},
                    {"range": [50, 100], "color": "#E9E1CC"}
                ],
            }
        ))
        fig_gauge.update_layout(
            height=200, margin=dict(l=20, r=20, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    card_class = "" if is_healthy else "disease"
    icon = "✅" if is_healthy else "⚠️"
    st.markdown(f"""
    <div class="tv-card {card_class}">
        <div class="tv-label">Teşhis</div>
        <h2>{icon} {info['tr_name']}</h2>
        <div class="tv-label">Açıklama</div>
        <p>{info['description']}</p>
        <div class="tv-label">Öneri</div>
        <p style="margin-bottom:0;">{info['suggestion']}</p>
    </div>
    """, unsafe_allow_html=True)

    if confidence < 50:
        st.warning("⚠️ **Sonuç kesin değil.** Farklı bir açıdan, daha net ve iyi ışıklandırılmış bir fotoğrafla tekrar deneyin.")

    st.markdown('<div class="tv-section-title">📊 Tüm sınıf olasılıkları</div>', unsafe_allow_html=True)

    sorted_pairs = sorted(zip(classes, prediction[0]), key=lambda x: x[1])
    sorted_labels = [disease_info[c]["tr_name"] for c, _ in sorted_pairs]
    sorted_values = [float(p) * 100 for _, p in sorted_pairs]
    bar_colors = ["#4F9D5C" if l == "Sağlıklı Yaprak" else "#C1432E" for l in sorted_labels]

    fig_bar = go.Figure(go.Bar(
        x=sorted_values, y=sorted_labels, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.2f}%" for v in sorted_values], textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11)
    ))
    fig_bar.update_layout(
        xaxis=dict(range=[0, 105], title="Olasılık (%)", gridcolor="#DDD5C2"),
        yaxis=dict(title=""),
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Work Sans")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("Ham veriyi gör"):
        for cls, prob in sorted(zip(classes, prediction[0]), key=lambda x: -x[1]):
            st.write(f"{disease_info[cls]['tr_name']}: {prob*100:.2f}%")

st.markdown('<div class="tv-footer">TomatoVision AI · EfficientNetB0 tabanlı derin öğrenme modeli</div>', unsafe_allow_html=True)
