import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import hashlib
import io
import gspread
from google.oauth2.service_account import Credentials

@st.cache_resource
def get_sheet():
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=scopes
        )
        gc = gspread.authorize(creds)
        sheet_name = st.secrets["sheet"]["name"]
        sh = gc.open(sheet_name)
        return sh.sheet1
    except Exception as e:
        st.sidebar.error(f"Sheets baglanti hatasi: {e}")
        return None


st.set_page_config(page_title="max01", page_icon="🍅", layout="centered")

# ---------- CSS (mobil uyumlu) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Work+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Work Sans', sans-serif; }
.stApp { background-color: #F6F2E9; color: #202B22; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}

.tv-banner {
    background: #1F3B2C;
    margin: 0 -1rem 1.5rem -1rem;
    padding: 2rem 1.5rem 1.5rem 1.5rem;
    border-bottom: 4px solid #C1432E;
}
.tv-banner h1 {
    font-family: 'Fraunces', serif; font-weight: 700;
    font-size: 2.4rem; color: #F6F2E9; margin: 0;
}
.tv-banner p { color: #B9C9BC; margin: 0.35rem 0 0 0; font-size: 1.02rem; }

[data-testid="stFileUploader"], [data-testid="stCameraInput"] {
    background: #EFE8D8; border: 2px dashed #4F9D5C;
    border-radius: 14px; padding: 1rem;
}

.tv-card {
    background: #EFE8D8; border-radius: 16px;
    padding: 1.5rem 1.75rem; margin-top: 1rem;
    border-left: 6px solid #4F9D5C;
}
.tv-card.disease { border-left: 6px solid #C1432E; }
.tv-card h2 { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.5rem; margin: 0 0 0.6rem 0; }
.tv-card .tv-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.08em; color: #5B6B5D; margin-bottom: 0.15rem;
}
.tv-card p { font-size: 0.97rem; line-height: 1.5; margin: 0.2rem 0 0.9rem 0; }

.tv-section-title {
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.25rem;
    margin: 1.75rem 0 0.5rem 0; color: #1F3B2C;
}
.tv-footer {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #8A968B;
    text-align: center; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #DDD5C2;
}

@media (max-width: 640px) {
    .tv-banner h1 { font-size: 1.7rem; }
    .tv-banner p { font-size: 0.88rem; }
    .tv-card { padding: 1.1rem 1.2rem; }
    .tv-card h2 { font-size: 1.2rem; }
}
</style>
""", unsafe_allow_html=True)

# ---------- Dil metinleri ----------
UI = {
    "tr": {
        "subtitle": "Domates Yaprağı Hastalık Tespiti",
        "upload_tab": "📁 Dosya Yükle",
        "camera_tab": "📷 Kamera",
        "upload_label": "🍅 Domates yaprağı fotoğrafı yükleyin",
        "camera_label": "Yaprağın fotoğrafını çekin",
        "diagnosis": "Teşhis", "description": "Açıklama",
        "suggestion": "Öneri", "medicine": "İlaç / Ürün Önerisi",
        "care": "Bakım Önerisi", "confidence": "Güven skoru",
        "low_conf": "⚠️ **Sonuç kesin değil.** Farklı bir açıdan, daha net ve iyi ışıklandırılmış bir fotoğrafla tekrar deneyin.",
        "all_classes": "📊 Tüm sınıf olasılıkları", "raw_data": "Ham veriyi gör",
        "history_title": "📊 Tahmin Geçmişi", "history_empty": "Henüz tahmin yapılmadı.",
        "clear_history": "Geçmişi temizle", "download_pdf": "📄 PDF olarak indir",
        "download_csv": "📊 Geçmişi CSV indir", "lang_label": "🌍 Dil / Language",
        "footer": "max01 · EfficientNetB0 tabanlı derin öğrenme modeli",
        "col_time": "Zaman", "col_class": "Sonuç", "col_conf": "Güven (%)"
    },
    "en": {
        "subtitle": "Tomato Leaf Disease Detection",
        "upload_tab": "📁 Upload File",
        "camera_tab": "📷 Camera",
        "upload_label": "🍅 Upload a tomato leaf photo",
        "camera_label": "Take a photo of the leaf",
        "diagnosis": "Diagnosis", "description": "Description",
        "suggestion": "Suggestion", "medicine": "Medicine / Product Suggestion",
        "care": "Care Tip", "confidence": "Confidence score",
        "low_conf": "⚠️ **Result is uncertain.** Try again with a clearer, better-lit photo from a different angle.",
        "all_classes": "📊 All class probabilities", "raw_data": "View raw data",
        "history_title": "📊 Prediction History", "history_empty": "No predictions yet.",
        "clear_history": "Clear history", "download_pdf": "📄 Download as PDF",
        "download_csv": "📊 Download history CSV", "lang_label": "🌍 Dil / Language",
        "footer": "max01 · Deep learning model based on EfficientNetB0",
        "col_time": "Time", "col_class": "Result", "col_conf": "Confidence (%)"
    }
}

classes = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

disease_info = {
    "Tomato_Bacterial_spot": {
        "name": {"tr": "Bakteriyel Leke Hastalığı", "en": "Bacterial Spot"},
        "description": {"tr": "Yapraklarda küçük, koyu kahverengi/siyah lekeler görülür. Nemli ve sıcak havalarda hızla yayılır.",
                         "en": "Small, dark brown/black spots appear on leaves. Spreads rapidly in warm, humid weather."},
        "suggestion": {"tr": "Sulama sırasında yaprakları ıslatmaktan kaçının, hastalıklı yaprakları uzaklaştırın.",
                        "en": "Avoid wetting leaves while watering, remove infected leaves."},
        "medicine": {"tr": "Bakır oksiklorür içerikli fungisit", "en": "Copper oxychloride-based fungicide"}
    },
    "Tomato_Early_blight": {
        "name": {"tr": "Erken Yanıklık (Alternaria)", "en": "Early Blight (Alternaria)"},
        "description": {"tr": "Yapraklarda konsantrik halkalı kahverengi lekeler oluşur, alt yapraklardan başlar.",
                         "en": "Concentric brown ring spots form on leaves, starting from lower leaves."},
        "suggestion": {"tr": "Bitki aralarını iyi havalandırın, düşük azotlu gübreleme yapın.",
                        "en": "Ensure good airflow between plants, use low-nitrogen fertilizer."},
        "medicine": {"tr": "Mancozeb veya klorotalonil içerikli fungisit", "en": "Mancozeb or chlorothalonil-based fungicide"}
    },
    "Tomato_Late_blight": {
        "name": {"tr": "Geç Yanıklık (Phytophthora)", "en": "Late Blight (Phytophthora)"},
        "description": {"tr": "Büyük, düzensiz, su emmiş görünümlü koyu lekeler oluşur. Çok hızlı yayılabilir.",
                         "en": "Large, irregular, water-soaked dark spots form. Can spread very quickly."},
        "suggestion": {"tr": "Etkilenen bitkileri hemen izole edin, nemli ortamlardan kaçının.",
                        "en": "Isolate affected plants immediately, avoid humid conditions."},
        "medicine": {"tr": "Metalaksil içerikli sistemik fungisit", "en": "Metalaxyl-based systemic fungicide"}
    },
    "Tomato_Leaf_Mold": {
        "name": {"tr": "Yaprak Küfü", "en": "Leaf Mold"},
        "description": {"tr": "Yaprak üstünde soluk sarı lekeler, altında kadifemsi zeytin-yeşili küf tabakası oluşur.",
                         "en": "Pale yellow spots on top of leaf, velvety olive-green mold underneath."},
        "suggestion": {"tr": "Nem oranını düşürün, havalandırmayı artırın.",
                        "en": "Reduce humidity, improve air circulation."},
        "medicine": {"tr": "Klorotalonil içerikli fungisit", "en": "Chlorothalonil-based fungicide"}
    },
    "Tomato_Septoria_leaf_spot": {
        "name": {"tr": "Septoria Yaprak Lekesi", "en": "Septoria Leaf Spot"},
        "description": {"tr": "Küçük, yuvarlak, gri merkezli ve koyu kenarlı lekeler yaprağı kaplar.",
                         "en": "Small, round spots with gray centers and dark edges cover the leaf."},
        "suggestion": {"tr": "Hastalıklı yaprakları toplayıp uzaklaştırın, sulamada yaprakları ıslatmayın.",
                        "en": "Remove infected leaves, avoid wetting leaves when watering."},
        "medicine": {"tr": "Bakır bazlı fungisit", "en": "Copper-based fungicide"}
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": {"tr": "Kırmızı Örümcek (İki Noktalı)", "en": "Two-Spotted Spider Mite"},
        "description": {"tr": "Yapraklarda sararma, noktalı desenler ve ince örümcek ağları görülür.",
                         "en": "Yellowing, speckled patterns, and fine webbing appear on leaves."},
        "suggestion": {"tr": "Bitkiyi düzenli nemlendirin, doğal düşmanlardan faydalanın.",
                        "en": "Keep the plant humid, encourage natural predators."},
        "medicine": {"tr": "Abamektin içerikli akarisit", "en": "Abamectin-based miticide"}
    },
    "Tomato_Target_Spot": {
        "name": {"tr": "Hedef Lekesi", "en": "Target Spot"},
        "description": {"tr": "Konsantrik halkalı, hedef tahtası görünümünde kahverengi lekeler oluşur.",
                         "en": "Brown spots with concentric rings, resembling a target, form on leaves."},
        "suggestion": {"tr": "Bitki artıklarını temizleyin, ekim nöbeti uygulayın.",
                        "en": "Clean up plant debris, practice crop rotation."},
        "medicine": {"tr": "Azoksistrobin içerikli fungisit", "en": "Azoxystrobin-based fungicide"}
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "name": {"tr": "Sarı Yaprak Kıvırcıklığı Virüsü", "en": "Yellow Leaf Curl Virus"},
        "description": {"tr": "Yapraklar sararır, küçülür ve kıvrılır. Beyazsinek tarafından taşınan viral bir hastalıktır.",
                         "en": "Leaves yellow, shrink, and curl. A viral disease spread by whiteflies."},
        "suggestion": {"tr": "Beyazsinek popülasyonunu kontrol altına alın, hastalıklı bitkileri sökün.",
                        "en": "Control whitefly population, remove infected plants."},
        "medicine": {"tr": "Beyazsinek için imidakloprid içerikli insektisit", "en": "Imidacloprid-based insecticide for whiteflies"}
    },
    "Tomato_Tomato_mosaic_virus": {
        "name": {"tr": "Mozaik Virüsü", "en": "Mosaic Virus"},
        "description": {"tr": "Yapraklarda açık-koyu yeşil mozaik desenli renk bozukluğu ve şekil bozulması görülür.",
                         "en": "Light-dark green mosaic discoloration and leaf deformation appear."},
        "suggestion": {"tr": "Hastalıklı bitkileri imha edin, alet-ekipmanı dezenfekte edin.",
                        "en": "Destroy infected plants, disinfect tools and equipment."},
        "medicine": {"tr": "Kimyasal tedavisi yoktur; önleyici hijyen önemlidir", "en": "No chemical cure; preventive hygiene is key"}
    },
    "Tomato_healthy": {
        "name": {"tr": "Sağlıklı Yaprak", "en": "Healthy Leaf"},
        "description": {"tr": "Yaprakta herhangi bir hastalık belirtisi tespit edilmedi.",
                         "en": "No disease symptoms were detected on the leaf."},
        "suggestion": {"tr": "Haftada 2-3 kez düzenli sulama yapın, güneşli bir noktada tutun, ayda bir organik gübre uygulayın, yaprakları düzenli kontrol edin.",
                        "en": "Water regularly 2-3 times a week, keep in a sunny spot, apply organic fertilizer monthly, inspect leaves regularly."},
        "medicine": {"tr": "-", "en": "-"}
    }
}

# ---------- Oturum durumu ----------
if "lang" not in st.session_state:
    st.session_state.lang = "tr"
if "history" not in st.session_state:
    st.session_state.history = []
if "last_hash" not in st.session_state:
    st.session_state.last_hash = None

# ---------- Sidebar: dil + geçmiş ----------
with st.sidebar:
    lang_choice = st.radio("🌍 Dil / Language", ["Türkçe", "English"],
                            index=0 if st.session_state.lang == "tr" else 1)
    st.session_state.lang = "tr" if lang_choice == "Türkçe" else "en"

L = UI[st.session_state.lang]

sheet = get_sheet()

with st.sidebar:
    st.markdown(f"### {L['history_title']}")
    sheet_rows = []
    if sheet is not None:
        try:
            sheet_rows = sheet.get_all_records()
        except Exception:
            sheet_rows = []
    if sheet_rows:
        df_hist = pd.DataFrame(sheet_rows)
        df_hist.columns = [L["col_time"], L["col_class"], L["col_conf"]]
        st.dataframe(df_hist.iloc[::-1], use_container_width=True, hide_index=True)
        csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(L["download_csv"], csv_bytes, "history.csv", "text/csv")
        if st.button(L["clear_history"]):
            sheet.clear()
            sheet.append_row(["time", "class", "conf"])
            st.rerun()
    else:
        st.caption(L["history_empty"])

# ---------- Banner ----------
st.markdown(f"""
<div class="tv-banner">
    <h1>🍅 max01</h1>
    <p>{L['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("TomatoVisionAI_EfficientNetB0.keras")

model = load_model()

# ---------- Görüntü kaynağı: dosya ya da kamera ----------
tab1, tab2 = st.tabs([L["upload_tab"], L["camera_tab"]])
uploaded_file = None
with tab1:
    f = st.file_uploader(L["upload_label"], type=["jpg", "jpeg", "png"])
    if f is not None:
        uploaded_file = f
with tab2:
    cam = st.camera_input(L["camera_label"])
    if cam is not None:
        uploaded_file = cam

def translit(text):
    m = str.maketrans("ğĞşŞıİçÇöÖüÜ", "gGsSiIcCoOuU")
    return text.translate(m)

def build_pdf(image, class_name_local, confidence, description, suggestion, medicine, lang):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    epw = pdf.w - 2 * pdf.l_margin

    pdf.set_font("Helvetica", "B", 16)
    title = "max01 - Rapor" if lang == "tr" else "max01 - Report"
    pdf.set_x(pdf.l_margin)
    pdf.cell(epw, 10, translit(title), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin)
    pdf.cell(epw, 8, datetime.now().strftime("%Y-%m-%d %H:%M"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    img_buf = io.BytesIO()
    image.convert("RGB").save(img_buf, format="JPEG")
    img_buf.seek(0)
    tmp_path = "/tmp/_report_img.jpg"
    with open(tmp_path, "wb") as fh:
        fh.write(img_buf.read())

    img_w_mm = 90
    aspect = image.height / image.width
    img_h_mm = img_w_mm * aspect
    pdf.image(tmp_path, x=pdf.l_margin, y=pdf.get_y(), w=img_w_mm)
    pdf.set_xy(pdf.l_margin, pdf.get_y() + img_h_mm + 6)

    label_diag = "Teshis" if lang == "tr" else "Diagnosis"
    label_conf = "Guven skoru" if lang == "tr" else "Confidence"
    label_desc = "Aciklama" if lang == "tr" else "Description"
    label_sugg = "Oneri" if lang == "tr" else "Suggestion"
    label_med = "Ilac / Urun" if lang == "tr" else "Medicine / Product"

    def safe_line(font_style, font_size, text, line_h=7):
        pdf.set_font("Helvetica", font_style, font_size)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(epw, line_h, text)
        pdf.ln(1)

    safe_line("B", 13, translit(f"{label_diag}: {class_name_local}"), 8)
    safe_line("", 11, f"{label_conf}: {confidence:.2f}%")
    safe_line("", 11, translit(f"{label_desc}: {description}"))
    safe_line("", 11, translit(f"{label_sugg}: {suggestion}"))
    safe_line("", 11, translit(f"{label_med}: {medicine}"))

    return bytes(pdf.output())

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_bytes_for_hash = uploaded_file.getvalue()
    current_hash = hashlib.md5(img_bytes_for_hash).hexdigest()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="", use_container_width=True)

    img_resized = image.resize((150, 150))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    with st.spinner("..."):
        prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction[0])
    predicted_class = classes[predicted_index]
    confidence = float(np.max(prediction[0])) * 100
    info = disease_info[predicted_class]
    lang = st.session_state.lang
    is_healthy = predicted_class == "Tomato_healthy"
    accent = "#4F9D5C" if is_healthy else "#C1432E"
    class_name_local = info["name"][lang]

    # Geçmişe ekle (aynı görüntü tekrar eklenmesin)
    if current_hash != st.session_state.last_hash:
        if sheet is not None:
            try:
                sheet.append_row([
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    class_name_local,
                    round(confidence, 2)
                ])
            except Exception as e:
                st.sidebar.warning(f"Kaydedilemedi: {e}")
        st.session_state.last_hash = current_hash

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=confidence,
            number={"suffix": "%", "font": {"family": "IBM Plex Mono", "size": 30, "color": "#202B22"}},
            gauge={"axis": {"range": [0, 100], "tickcolor": "#8A968B"},
                   "bar": {"color": accent}, "bgcolor": "#EFE8D8", "borderwidth": 0,
                   "steps": [{"range": [0, 50], "color": "#E4DCC8"}, {"range": [50, 100], "color": "#E9E1CC"}]}
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    card_class = "" if is_healthy else "disease"
    icon = "✅" if is_healthy else "⚠️"
    suggestion_label = L["care"] if is_healthy else L["suggestion"]
    st.markdown(f"""
    <div class="tv-card {card_class}">
        <div class="tv-label">{L['diagnosis']}</div>
        <h2>{icon} {class_name_local}</h2>
        <div class="tv-label">{L['description']}</div>
        <p>{info['description'][lang]}</p>
        <div class="tv-label">{suggestion_label}</div>
        <p>{info['suggestion'][lang]}</p>
        <div class="tv-label">{L['medicine']}</div>
        <p style="margin-bottom:0;">{info['medicine'][lang]}</p>
    </div>
    """, unsafe_allow_html=True)

    if confidence < 50:
        st.warning(L["low_conf"])

    pdf_bytes = build_pdf(image, class_name_local, confidence,
                           info['description'][lang], info['suggestion'][lang],
                           info['medicine'][lang], lang)
    st.download_button(L["download_pdf"], pdf_bytes, "max01_rapor.pdf", "application/pdf")

    st.markdown(f'<div class="tv-section-title">{L["all_classes"]}</div>', unsafe_allow_html=True)

    sorted_pairs = sorted(zip(classes, prediction[0]), key=lambda x: x[1])
    sorted_labels = [disease_info[c]["name"][lang] for c, _ in sorted_pairs]
    sorted_values = [float(p) * 100 for _, p in sorted_pairs]
    bar_colors = ["#4F9D5C" if c == "Tomato_healthy" else "#C1432E" for c, _ in sorted_pairs]

    fig_bar = go.Figure(go.Bar(
        x=sorted_values, y=sorted_labels, orientation="h", marker_color=bar_colors,
        text=[f"{v:.2f}%" for v in sorted_values], textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11)
    ))
    fig_bar.update_layout(
        xaxis=dict(range=[0, 105], title="%", gridcolor="#DDD5C2"), yaxis=dict(title=""),
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Work Sans")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander(L["raw_data"]):
        for cls, prob in sorted(zip(classes, prediction[0]), key=lambda x: -x[1]):
            st.write(f"{disease_info[cls]['name'][lang]}: {prob*100:.2f}%")

st.markdown(f'<div class="tv-footer">{L["footer"]}</div>', unsafe_allow_html=True)
