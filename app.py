import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime
import hashlib
import io
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="max01", page_icon="🍅", layout="centered")

# ================= CSS: Koyu tema + Glassmorphism =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Work+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Work Sans', sans-serif; }

.stApp {
    background: radial-gradient(circle at 20% 0%, #16241C 0%, #0D1410 60%);
    color: #EAF2EC;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.tv-hero {
    background: linear-gradient(135deg, rgba(31,59,44,0.85), rgba(13,20,16,0.85));
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 22px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.tv-hero .emoji-row { font-size: 2.2rem; margin-bottom: 0.4rem; }
.tv-hero h1 {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.3rem;
    background: linear-gradient(90deg, #7FDB8F, #F2C94C);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.tv-hero p { color: #C7D6CC; margin: 0.5rem 0 1.5rem 0; font-size: 1.02rem; }

.tv-glass {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.tv-glass.disease { border-left: 4px solid #E4573D; }
.tv-glass.healthy { border-left: 4px solid #4FCB6B; }

.tv-glass h2 {
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.4rem;
    margin: 0 0 0.6rem 0; color: #F3F7F4;
}
.tv-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.08em; color: #8FAE9A; margin: 0.7rem 0 0.15rem 0;
}
.tv-glass p { font-size: 0.95rem; line-height: 1.55; margin: 0.1rem 0; color: #DCE7DF; }

.risk-badge {
    display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 600;
    margin-top: 0.3rem;
}
.risk-low { background: rgba(79,203,107,0.18); color: #7FDB8F; border: 1px solid #4FCB6B; }
.risk-mid { background: rgba(242,201,76,0.18); color: #F2C94C; border: 1px solid #F2C94C; }
.risk-high { background: rgba(228,87,61,0.18); color: #F08A72; border: 1px solid #E4573D; }

.tv-section-title {
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.2rem;
    margin: 1.6rem 0 0.5rem 0; color: #EAF2EC;
}

.top3-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.07);
    font-size: 0.92rem;
}

.about-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px; padding: 1.6rem; text-align: center;
}

[data-testid="stFileUploader"], [data-testid="stCameraInput"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed #4FCB6B; border-radius: 14px; padding: 1rem;
}

.tv-footer {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #7C8F82;
    text-align: center; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.08);
}

@media (max-width: 640px) {
    .tv-hero h1 { font-size: 1.6rem; }
    .tv-hero p { font-size: 0.85rem; }
    .tv-glass { padding: 1.1rem 1.2rem; }
}
</style>
""", unsafe_allow_html=True)

# ================= Google Sheets bağlantısı =================
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
        st.sidebar.error(f"Sheets baglanti hatasi: {type(e).__name__}")
        return None

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

# Risk seviyesi (hastalığın genel ciddiyetine göre, sabit)
risk_level = {
    "Tomato_Bacterial_spot": "mid",
    "Tomato_Early_blight": "mid",
    "Tomato_Late_blight": "high",
    "Tomato_Leaf_Mold": "mid",
    "Tomato_Septoria_leaf_spot": "mid",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "mid",
    "Tomato_Target_Spot": "mid",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": "high",
    "Tomato_Tomato_mosaic_virus": "high",
    "Tomato_healthy": "low",
}

# ================= Dil metinleri (arayüz) =================
UI = {
    "tr": {
        "nav": ["🏠 Ana Sayfa", "📷 Yaprak Analizi", "🌱 Hastalık Bilgisi", "📍 Tarım Desteği", "👨‍💻 Hakkında"],
        "hero_title": "TomatoVision AI",
        "hero_sub": "Yapay Zeka Destekli Domates Yaprağı Hastalık Tespiti",
        "start_btn": "Analize Başla →",
        "upload_tab": "📁 Dosya Yükle", "camera_tab": "📷 Kamera",
        "upload_label": "🍅 Domates yaprağı fotoğrafı yükleyin",
        "camera_label": "Yaprağın fotoğrafını çekin",
        "diagnosis": "Teşhis", "description": "Açıklama", "symptoms": "Belirtiler",
        "pathogen": "Etken", "spread": "Yayılma Koşulları",
        "treatment": "Mücadele Yöntemi", "prevention": "Korunma Önerisi",
        "care": "Bakım Önerisi", "confidence": "Güven oranı", "risk": "Risk Seviyesi",
        "risk_low": "🟢 Düşük", "risk_mid": "🟡 Orta", "risk_high": "🔴 Yüksek",
        "reliability_note": "Bu sonuç modelin görüntüden çıkardığı istatistiksel bir tahmindir; kesin teşhis için bir ziraat mühendisine danışmanız önerilir.",
        "low_conf": "⚠️ Sonuç kesin değil. Farklı bir açıdan, net ve iyi ışıklandırılmış bir fotoğrafla tekrar deneyin.",
        "top3": "🏆 En Yüksek 3 Tahmin",
        "all_classes": "📊 Tüm sınıf olasılıkları", "raw_data": "Ham veriyi gör",
        "history_title": "📊 Tahmin Geçmişi", "history_empty": "Henüz tahmin yapılmadı.",
        "clear_history": "Geçmişi temizle", "download_pdf": "📄 AI Raporu indir (PDF)",
        "download_csv": "📊 Geçmişi CSV indir", "lang_label": "🌍 Dil / Language",
        "footer": "max01 · EfficientNetB0 tabanlı derin öğrenme modeli",
        "col_time": "Zaman", "col_class": "Sonuç", "col_conf": "Güven (%)",
        "disease_db_title": "🌱 Hastalık Bilgi Bankası",
        "disease_db_sub": "Aşağıdan bir hastalık seçerek detaylı bilgi alabilirsiniz.",
        "support_title": "📍 Tarım Desteği",
        "support_sub": "Bölgenizi seçin, size uygun resmi destek kanallarını gösterelim.",
        "support_region_label": "Bölgenizi seçin",
        "support_generic": "Bölgenizdeki en güncel ve doğru bilgi için resmi kaynaklara başvurmanızı öneririz:",
        "support_line1": "📞 ALO 180 Tarım ve Orman Bakanlığı Danışma Hattı",
        "support_line2": "🏢 İlçenizdeki Tarım ve Orman İl/İlçe Müdürlüğü",
        "support_line3": "👨‍🌾 Ziraat Mühendisleri Odası temsilciliği",
        "support_note": "Not: Bu bölüm, size en yakın gerçek zamanlı bayii/danışman konumunu göstermez; genel resmi yönlendirme sağlar.",
        "about_title": "TomatoVision AI",
        "about_dev": "Geliştirici: Mohamed Ahmed Hassan",
        "about_uni": "Çukurova Üniversitesi",
        "about_dept": "Tarım Makineleri ve Teknolojileri Mühendisliği",
        "about_stack": "Deep Learning • EfficientNetB0 • Streamlit",
    },
    "en": {
        "nav": ["🏠 Home", "📷 Leaf Analysis", "🌱 Disease Info", "📍 Agri Support", "👨‍💻 About"],
        "hero_title": "TomatoVision AI",
        "hero_sub": "AI Powered Tomato Leaf Disease Detection",
        "start_btn": "Start Diagnosis →",
        "upload_tab": "📁 Upload File", "camera_tab": "📷 Camera",
        "upload_label": "🍅 Upload a tomato leaf photo",
        "camera_label": "Take a photo of the leaf",
        "diagnosis": "Diagnosis", "description": "Description", "symptoms": "Symptoms",
        "pathogen": "Pathogen", "spread": "Spread Conditions",
        "treatment": "Treatment", "prevention": "Prevention",
        "care": "Care Tip", "confidence": "Confidence", "risk": "Risk Level",
        "risk_low": "🟢 Low", "risk_mid": "🟡 Medium", "risk_high": "🔴 High",
        "reliability_note": "This result is a statistical estimate from the model; consult an agricultural engineer for a definitive diagnosis.",
        "low_conf": "⚠️ Result is uncertain. Try again with a clearer, better-lit photo from a different angle.",
        "top3": "🏆 Top 3 Predictions",
        "all_classes": "📊 All class probabilities", "raw_data": "View raw data",
        "history_title": "📊 Prediction History", "history_empty": "No predictions yet.",
        "clear_history": "Clear history", "download_pdf": "📄 Download AI Report (PDF)",
        "download_csv": "📊 Download history CSV", "lang_label": "🌍 Dil / Language",
        "footer": "max01 · Deep learning model based on EfficientNetB0",
        "col_time": "Time", "col_class": "Result", "col_conf": "Confidence (%)",
        "disease_db_title": "🌱 Disease Knowledge Base",
        "disease_db_sub": "Select a disease below for detailed information.",
        "support_title": "📍 Agricultural Support",
        "support_sub": "Select your region to see relevant official support channels.",
        "support_region_label": "Select your region",
        "support_generic": "For the most accurate, up-to-date info in your area, please contact official sources:",
        "support_line1": "📞 ALO 180 Ministry of Agriculture Advisory Line (Turkey)",
        "support_line2": "🏢 Your local Provincial/District Agriculture Directorate",
        "support_line3": "👨‍🌾 Chamber of Agricultural Engineers, local representation",
        "support_note": "Note: This section does not show real-time nearby dealers/advisors; it provides general official guidance.",
        "about_title": "TomatoVision AI",
        "about_dev": "Developed by: Mohamed Ahmed Hassan",
        "about_uni": "Çukurova University",
        "about_dept": "Agricultural Machinery and Technologies Engineering",
        "about_stack": "Deep Learning • EfficientNetB0 • Streamlit",
    },
    "so": {
        "nav": ["🏠 Bogga Hore", "📷 Falanqaynta Caleenta", "🌱 Macluumaadka Cudurka", "📍 Taageerada Beeraha", "👨‍💻 Ku Saabsan"],
        "hero_title": "TomatoVision AI",
        "hero_sub": "Ogaanshaha Cudurrada Caleenta Yaanyada ee AI-ga",
        "start_btn": "Bilow Baaritaanka →",
        "upload_tab": "📁 Soo Geli Sawir", "camera_tab": "📷 Kamarad",
        "upload_label": "🍅 Soo geli sawirka caleenta yaanyada",
        "camera_label": "Qaado sawir caleenta",
        "diagnosis": "Baaritaan", "description": "Sharraxaad", "symptoms": "Calaamado",
        "pathogen": "Sababta", "spread": "Xaaladaha Faafitaanka",
        "treatment": "Daawaynta", "prevention": "Ka Hortagga",
        "care": "Talooyin Daryeel", "confidence": "Kalsoonida", "risk": "Heerka Khatarta",
        "risk_low": "🟢 Hooseeya", "risk_mid": "🟡 Dhexdhexaad", "risk_high": "🔴 Sareeya",
        "reliability_note": "Natiijadan waa qiyaas tirakoob oo ka yimid moodelka; fadlan la tashii injineer beeraha si aad u hesho baaritaan sax ah.",
        "low_conf": "⚠️ Natiijadu ma hubna. Isku day mar kale sawir cad oo iftiin fiican leh.",
        "top3": "🏆 3-da Ugu Sarreeya",
        "all_classes": "📊 Dhammaan itixaadka fasallada", "raw_data": "Fiiri xogta cad",
        "history_title": "📊 Taariikhda Baaritaannada", "history_empty": "Wali lama sameyn baaritaan.",
        "clear_history": "Nadiifi taariikhda", "download_pdf": "📄 Soo deji Warbixinta AI (PDF)",
        "download_csv": "📊 Soo deji Taariikhda CSV", "lang_label": "🌍 Luqadda / Language",
        "footer": "max01 · Moodel wax-baris oo ku salaysan EfficientNetB0",
        "col_time": "Waqti", "col_class": "Natiijo", "col_conf": "Kalsooni (%)",
        "disease_db_title": "🌱 Kaydka Macluumaadka Cudurrada",
        "disease_db_sub": "Ka dooro cudur si aad u hesho macluumaad faahfaahsan.",
        "support_title": "📍 Taageerada Beeraha",
        "support_sub": "Dooro gobolkaaga si aan kuu tusno kanaalada taageerada rasmiga ah.",
        "support_region_label": "Dooro gobolkaaga",
        "support_generic": "Macluumaadka ugu sax badan ee gobolkaaga, fadlan la xiriir ilaha rasmiga ah:",
        "support_line1": "📞 Khadka Talobixinta Wasaaradda Beeraha (Turkiga) ALO 180",
        "support_line2": "🏢 Xafiiska Beeraha ee Gobolkaaga/Degmadaada",
        "support_line3": "👨‍🌾 Wakiilka Golaha Injineerada Beeraha",
        "support_note": "Fiiro gaar ah: Qaybtan ma tusayo goobaha suuqyada/la-taliyeyaasha kuu dhow ee waqtiga dhabta ah; waxay bixisaa hagitaan guud oo rasmi ah.",
        "about_title": "TomatoVision AI",
        "about_dev": "Waxaa sameeyay: Mohamed Ahmed Hassan",
        "about_uni": "Jaamacadda Çukurova",
        "about_dept": "Injineeriyadda Qalabka iyo Tignoolajiyada Beeraha",
        "about_stack": "Deep Learning • EfficientNetB0 • Streamlit",
    }
}

# ================= Hastalık veritabanı (genişletilmiş) =================
disease_info = {
    "Tomato_Bacterial_spot": {
        "name": {"tr": "Bakteriyel Leke Hastalığı", "en": "Bacterial Spot", "so": "Cudurka Bakteeriyada"},
        "description": {"tr": "Yapraklarda küçük, koyu kahverengi/siyah lekeler oluşturan yaygın bir bakteriyel hastalıktır.",
                        "en": "A common bacterial disease causing small, dark brown/black spots on leaves.",
                        "so": "Cudur bakteeriya ah oo ka dhalinaya bar madow yar oo caleenta ku yaal."},
        "symptoms": {"tr": "Küçük, sulu görünümlü, koyu kahverengi-siyah lekeler; ileri aşamada yaprak dökülmesi.",
                     "en": "Small, water-soaked, dark brown-black spots; leaf drop in advanced stages.",
                     "so": "Bararka yaryar ee madow, marka uu sii xumaado caleenta way dhacdaa."},
        "pathogen": {"tr": "Bakteri (Xanthomonas türleri)", "en": "Bacteria (Xanthomonas spp.)", "so": "Bakteeriya (Xanthomonas)"},
        "spread": {"tr": "Sıcak, nemli hava ve yaprak ıslaklığı ile hızla yayılır; sulama sırasında sıçrayan su ile bulaşır.",
                   "en": "Spreads rapidly in warm, humid weather and leaf wetness; transmitted via splashing water.",
                   "so": "Waxay si dhaqso ah ugu faafaa hawada kulul ee qoyan; waxay ku faafaan biyaha lagu shubo."},
        "treatment": {"tr": "Bakır oksiklorür içerikli fungisit/bakterisit uygulamaları düzenli aralıklarla yapılmalıdır.",
                      "en": "Apply copper oxychloride-based bactericide/fungicide at regular intervals.",
                      "so": "Isticmaal daawooyinka koorside ku salaysan si joogto ah."},
        "prevention": {"tr": "Sertifikalı tohum kullanın, yaprakları ıslatmadan sulayın, bitki artıklarını temizleyin.",
                       "en": "Use certified seed, avoid wetting leaves during irrigation, remove plant debris.",
                       "so": "Isticmaal iniinta la xaqiijiyay, ha qoynin caleemaha, nadiifi hadhaaga dhirta."},
    },
    "Tomato_Early_blight": {
        "name": {"tr": "Erken Yanıklık (Alternaria)", "en": "Early Blight (Alternaria)", "so": "Gubasho Hore (Alternaria)"},
        "description": {"tr": "Genellikle alt yapraklardan başlayan, konsantrik halkalı kahverengi lekelerle karakterize bir mantar hastalığıdır.",
                         "en": "A fungal disease characterized by concentric-ringed brown spots, usually starting on lower leaves.",
                         "so": "Cudur fangas ah oo ka bilaabma caleemaha hoose."},
        "symptoms": {"tr": "Hedef tahtası görünümünde halkalı kahverengi lekeler, yaprak sararması ve dökülmesi.",
                     "en": "Target-like concentric brown rings, yellowing and leaf drop.",
                     "so": "Bararka madow ee giraan-giraan ah, huruud iyo dhicid caleemo."},
        "pathogen": {"tr": "Mantar (Alternaria solani)", "en": "Fungus (Alternaria solani)", "so": "Fangas (Alternaria solani)"},
        "spread": {"tr": "Sıcak (24-29°C), nemli hava koşullarında ve stresli bitkilerde daha sık görülür.",
                   "en": "More common in warm (24-29°C), humid conditions and stressed plants.",
                   "so": "Waxay ku badataa xilliyada kulul ee qoyan."},
        "treatment": {"tr": "Mancozeb veya klorotalonil içerikli fungisitlerle 7-10 günde bir mücadele edilir.",
                      "en": "Treat every 7-10 days with mancozeb or chlorothalonil-based fungicides.",
                      "so": "Isticmaal fangaskiile 7-10 maalin mar."},
        "prevention": {"tr": "Bitki aralarını iyi havalandırın, düşük azotlu dengeli gübreleme yapın, ekim nöbeti uygulayın.",
                       "en": "Ensure good airflow, use balanced low-nitrogen fertilization, practice crop rotation.",
                       "so": "Siin hawo ku filan dhirta, isticmaal bacriminta la miisaamay."},
    },
    "Tomato_Late_blight": {
        "name": {"tr": "Geç Yanıklık (Phytophthora)", "en": "Late Blight (Phytophthora)", "so": "Gubasho Dambe (Phytophthora)"},
        "description": {"tr": "Çok hızlı yayılan, ciddi ürün kayıplarına yol açabilen yıkıcı bir mantar hastalığıdır.",
                         "en": "A devastating fungal disease that spreads very fast and can cause severe crop loss.",
                         "so": "Cudur fangas ah oo si xoog leh u faafa oo khasaare weyn keena."},
        "symptoms": {"tr": "Büyük, düzensiz, su emmiş görünümlü koyu yeşil-kahverengi lekeler; nemli havada beyazımsı küf tabakası.",
                     "en": "Large, irregular, water-soaked dark green-brown lesions; whitish mold in humid weather.",
                     "so": "Bararka waaweyn ee madow-cagaaran, marka qoyan noqoto waxaa ka soo baxa fatoora cad."},
        "pathogen": {"tr": "Su küfü (Phytophthora infestans)", "en": "Oomycete (Phytophthora infestans)", "so": "Fangas biyood (Phytophthora infestans)"},
        "spread": {"tr": "Serin ve nemli havada (15-20°C) çok hızlı yayılır; birkaç gün içinde tüm tarlayı etkileyebilir.",
                   "en": "Spreads extremely fast in cool, humid weather (15-20°C); can affect an entire field within days.",
                   "so": "Waxay si dhaqso ah ugu faafaa hawada qabow ee qoyan."},
        "treatment": {"tr": "Metalaksil içerikli sistemik fungisit acilen uygulanmalı, etkilenen bitkiler izole edilmelidir.",
                      "en": "Apply metalaxyl-based systemic fungicide immediately; isolate affected plants.",
                      "so": "Isticmaal daawo si degdeg ah, kala saar dhirta la qabo."},
        "prevention": {"tr": "Dayanıklı çeşitler seçin, nemli ortamlardan kaçının, düzenli tarla kontrolü yapın.",
                       "en": "Choose resistant varieties, avoid humid conditions, inspect fields regularly.",
                       "so": "Dooro noocyada iska caabiya, ka fogow qoyaanka, kormeer joogto ah samee."},
    },
    "Tomato_Leaf_Mold": {
        "name": {"tr": "Yaprak Küfü", "en": "Leaf Mold", "so": "Fatoorka Caleenta"},
        "description": {"tr": "Özellikle sera koşullarında, yüksek nemde görülen bir mantar hastalığıdır.",
                         "en": "A fungal disease common in high humidity, especially in greenhouse conditions.",
                         "so": "Cudur fangas ah oo ku badan gudaha beeraha saqafka leh."},
        "symptoms": {"tr": "Yaprak üstünde soluk sarı lekeler, altında kadifemsi zeytin-yeşili küf tabakası.",
                     "en": "Pale yellow spots on top of leaf, velvety olive-green mold underneath.",
                     "so": "Bararka jaalaha caleenta korkeeda, hoosteedana fatoor cagaaran leh."},
        "pathogen": {"tr": "Mantar (Passalora fulva)", "en": "Fungus (Passalora fulva)", "so": "Fangas (Passalora fulva)"},
        "spread": {"tr": "Yüksek nem (>%85) ve zayıf havalandırma ile hızla yayılır.",
                   "en": "Spreads fast with high humidity (>85%) and poor ventilation.",
                   "so": "Waxay ku faafaan qoyaanka sare iyo hawo xun."},
        "treatment": {"tr": "Klorotalonil içerikli fungisit uygulayın, nemi düşürün.",
                      "en": "Apply chlorothalonil-based fungicide, reduce humidity.",
                      "so": "Isticmaal fangaskiile, yaree qoyaanka."},
        "prevention": {"tr": "Sera havalandırmasını artırın, bitki sıklığını azaltın, damla sulama tercih edin.",
                       "en": "Improve greenhouse ventilation, reduce plant density, prefer drip irrigation.",
                       "so": "Hagaaji hawada beerta, yaree cufnaanta dhirta."},
    },
    "Tomato_Septoria_leaf_spot": {
        "name": {"tr": "Septoria Yaprak Lekesi", "en": "Septoria Leaf Spot", "so": "Baraha Caleenta Septoria"},
        "description": {"tr": "Küçük, gri merkezli lekelerle karakterize, yaygın bir mantar hastalığıdır.",
                         "en": "A common fungal disease characterized by small, gray-centered spots.",
                         "so": "Cudur fangas ah oo leh barar yaryar oo dhexdooda cirro ah."},
        "symptoms": {"tr": "Yuvarlak, gri merkezli, koyu kenarlı küçük lekeler; şiddetli enfeksiyonda yaprak dökülmesi.",
                     "en": "Small round spots with gray centers and dark margins; leaf drop in severe cases.",
                     "so": "Barar yaryar oo dhexdooda cirro leh, geesahoodana madow yihiin."},
        "pathogen": {"tr": "Mantar (Septoria lycopersici)", "en": "Fungus (Septoria lycopersici)", "so": "Fangas (Septoria lycopersici)"},
        "spread": {"tr": "Nemli hava ve sıçrayan su damlacıklarıyla yayılır, alt yapraklardan başlar.",
                   "en": "Spreads via humid weather and splashing water, starts on lower leaves.",
                   "so": "Waxay ku faafaan qoyaanka iyo biyaha lagu shubo."},
        "treatment": {"tr": "Bakır bazlı fungisit uygulanmalı, hastalıklı yapraklar toplanmalıdır.",
                      "en": "Apply copper-based fungicide, remove infected leaves.",
                      "so": "Isticmaal fangaskiile koorid ku salaysan."},
        "prevention": {"tr": "Sulamada yaprakları ıslatmaktan kaçının, bitki artıklarını temizleyin.",
                       "en": "Avoid wetting leaves when watering, clean up plant debris.",
                       "so": "Ha qoynin caleemaha marka aad waraabinayso."},
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": {"tr": "Kırmızı Örümcek (İki Noktalı)", "en": "Two-Spotted Spider Mite", "so": "Caaryaha Labo-Bar"},
        "description": {"tr": "Yaprak özsuyu emerek zarar veren, kuru ve sıcak havada hızla çoğalan bir zararlıdır.",
                         "en": "A pest that feeds on leaf sap and multiplies rapidly in dry, hot weather.",
                         "so": "Cayayaan cunaya dheecaanka caleenta, ku badan hawada qallalan ee kulul."},
        "symptoms": {"tr": "Yapraklarda sararma, noktalı desenler ve ince örümcek ağları.",
                     "en": "Yellowing, speckled patterns, and fine webbing on leaves.",
                     "so": "Caleenta way huruudaysaa, waxaana ka muuqda xarig caaryo ah."},
        "pathogen": {"tr": "Zararlı böcek (akar)", "en": "Pest (mite)", "so": "Cayayaan (caaryo)"},
        "spread": {"tr": "Kuru, sıcak havada ve su stresi altındaki bitkilerde hızla artar.",
                   "en": "Increases rapidly in dry, hot weather and water-stressed plants.",
                   "so": "Waxay ku badataa hawada qallalan ee kulul."},
        "treatment": {"tr": "Abamektin içerikli akarisit uygulayın, bitkiyi düzenli nemlendirin.",
                      "en": "Apply abamectin-based miticide, keep the plant humid.",
                      "so": "Isticmaal caaryo-dile, joogtee qoyaanka dhirta."},
        "prevention": {"tr": "Düzenli su verin, doğal düşmanlardan (uğur böceği vb.) faydalanın.",
                       "en": "Water regularly, encourage natural predators (ladybugs, etc.).",
                       "so": "Waraabi si joogto ah, isticmaal cayayaanka faa'iidada leh."},
    },
    "Tomato_Target_Spot": {
        "name": {"tr": "Hedef Lekesi", "en": "Target Spot", "so": "Baraha Bartilmaameedka"},
        "description": {"tr": "Hedef tahtası görünümünde lekeler oluşturan bir mantar hastalığıdır.",
                         "en": "A fungal disease that forms target-board-like ring spots.",
                         "so": "Cudur fangas ah oo samaynaya barar giraan-giraan ah."},
        "symptoms": {"tr": "Konsantrik halkalı, kahverengi, hedef tahtası görünümünde lekeler.",
                     "en": "Concentric-ringed brown spots resembling a target.",
                     "so": "Barar madow oo giraan-giraan ah."},
        "pathogen": {"tr": "Mantar (Corynespora cassiicola)", "en": "Fungus (Corynespora cassiicola)", "so": "Fangas (Corynespora cassiicola)"},
        "spread": {"tr": "Sıcak, nemli havada ve yoğun bitki örtüsünde hızla yayılır.",
                   "en": "Spreads rapidly in warm, humid weather and dense canopy.",
                   "so": "Waxay ku faafaan hawada kulul ee qoyan."},
        "treatment": {"tr": "Azoksistrobin içerikli fungisit uygulayın.",
                      "en": "Apply azoxystrobin-based fungicide.",
                      "so": "Isticmaal fangaskiile azoxystrobin ah."},
        "prevention": {"tr": "Bitki artıklarını temizleyin, ekim nöbeti uygulayın.",
                       "en": "Clean up plant debris, practice crop rotation.",
                       "so": "Nadiifi hadhaaga dhirta, bedbeddel dalagyada."},
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "name": {"tr": "Sarı Yaprak Kıvırcıklığı Virüsü", "en": "Yellow Leaf Curl Virus", "so": "Fayraska Caleenta Jaalaha Ah"},
        "description": {"tr": "Beyazsinek tarafından taşınan, ciddi verim kaybına yol açan viral bir hastalıktır.",
                         "en": "A viral disease transmitted by whiteflies, causing severe yield loss.",
                         "so": "Fayras ay sitaan duqsiga cad, wuxuuna keenaa khasaare weyn."},
        "symptoms": {"tr": "Yapraklarda sararma, küçülme, kıvrılma ve bitkide bodurlaşma.",
                     "en": "Leaf yellowing, shrinking, curling, and plant stunting.",
                     "so": "Caleenta way huruudaysaa, yaraataa, gudubtaana, dhirtuna way koraan la'yihiin."},
        "pathogen": {"tr": "Virüs (Begomovirus) - Beyazsinek ile taşınır", "en": "Virus (Begomovirus) - transmitted by whitefly", "so": "Fayras - waxaa sida duqsiga cad"},
        "spread": {"tr": "Beyazsinek popülasyonunun yoğun olduğu sıcak bölgelerde hızla yayılır.",
                   "en": "Spreads rapidly in warm regions with high whitefly populations.",
                   "so": "Waxay ku faafaan meelaha kulul ee duqsi badan."},
        "treatment": {"tr": "Kimyasal tedavisi yoktur; beyazsinek için imidakloprid içerikli insektisit kullanılır.",
                      "en": "No chemical cure; use imidacloprid-based insecticide to control whiteflies.",
                      "so": "Daawo kiimikaad lama heli karo; isticmaal cayayaan-dile si aad u xakameyso duqsiga."},
        "prevention": {"tr": "Beyazsinek popülasyonunu kontrol edin, hastalıklı bitkileri sökün, dayanıklı çeşitler kullanın.",
                       "en": "Control whitefly population, remove infected plants, use resistant varieties.",
                       "so": "Xakamee duqsiga cad, ka saar dhirta xanuunsan, isticmaal noocyo iska caabiya."},
    },
    "Tomato_Tomato_mosaic_virus": {
        "name": {"tr": "Mozaik Virüsü", "en": "Mosaic Virus", "so": "Fayraska Mosaic"},
        "description": {"tr": "Yaprakta desenli renk bozukluğuna ve şekil bozulmasına yol açan viral bir hastalıktır.",
                         "en": "A viral disease causing mottled discoloration and leaf deformation.",
                         "so": "Fayras keena isbedel midab caleenta iyo qaab dhismeedkeeda."},
        "symptoms": {"tr": "Açık-koyu yeşil mozaik desenli renk bozukluğu, yaprakta kıvrılma ve küçülme.",
                     "en": "Light-dark green mosaic discoloration, leaf curling and shrinking.",
                     "so": "Midab isbeddel cagaaran leh, caleentu way gudubtaa oo yaraataa."},
        "pathogen": {"tr": "Virüs (Tobamovirus)", "en": "Virus (Tobamovirus)", "so": "Fayras (Tobamovirus)"},
        "spread": {"tr": "Alet-ekipman, el teması ve tohum yoluyla kolayca bulaşır.",
                   "en": "Easily transmitted via tools, hand contact, and seed.",
                   "so": "Waxay ku faafaan qalabka, taabashada gacanta iyo iniinta."},
        "treatment": {"tr": "Kimyasal tedavisi yoktur; hastalıklı bitkiler imha edilmelidir.",
                      "en": "No chemical cure; infected plants should be destroyed.",
                      "so": "Daawo kiimikaad lama heli karo; dhirta xanuunsan waa in la baabi'iyaa."},
        "prevention": {"tr": "Alet-ekipmanı dezenfekte edin, sağlıklı sertifikalı tohum/fide kullanın.",
                       "en": "Disinfect tools and equipment, use healthy certified seed/seedlings.",
                       "so": "Nadiifi qalabka, isticmaal iniin caafimaad qaba."},
    },
    "Tomato_healthy": {
        "name": {"tr": "Sağlıklı Yaprak", "en": "Healthy Leaf", "so": "Caleen Caafimaad Qabta"},
        "description": {"tr": "Yaprakta herhangi bir hastalık belirtisi tespit edilmedi.",
                         "en": "No disease symptoms were detected on the leaf.",
                         "so": "Wax calaamado cudur ah lagama helin caleenta."},
        "symptoms": {"tr": "-", "en": "-", "so": "-"},
        "pathogen": {"tr": "-", "en": "-", "so": "-"},
        "spread": {"tr": "-", "en": "-", "so": "-"},
        "treatment": {"tr": "-", "en": "-", "so": "-"},
        "prevention": {"tr": "Haftada 2-3 kez düzenli sulama yapın, güneşli bir noktada tutun, ayda bir organik gübre uygulayın, yaprakları düzenli kontrol edin.",
                       "en": "Water regularly 2-3 times a week, keep in a sunny spot, apply organic fertilizer monthly, inspect leaves regularly.",
                       "so": "Waraabi 2-3 jeer usbuucii, geli meel qorraxda leh, isticmaal bacrin dabiici ah bishii mar."},
    }
}

# ================= Bölgeler (Tarım Desteği için) =================
regions = {
    "tr": ["Marmara", "Ege", "Akdeniz", "İç Anadolu", "Karadeniz", "Doğu Anadolu", "Güneydoğu Anadolu"],
    "en": ["Marmara", "Aegean", "Mediterranean", "Central Anatolia", "Black Sea", "Eastern Anatolia", "Southeastern Anatolia"],
    "so": ["Marmara", "Aegean", "Baddda Dhexe", "Anatolia Dhexe", "Badda Madow", "Anatolia Bari", "Anatolia Koonfur-Bari"],
}

# ================= Oturum durumu =================
if "lang" not in st.session_state:
    st.session_state.lang = "tr"
if "page" not in st.session_state:
    st.session_state.page = 0
if "last_hash" not in st.session_state:
    st.session_state.last_hash = None

lang_names = {"tr": "🇹🇷 Türkçe", "en": "🇬🇧 English", "so": "🇸🇴 Somali"}
with st.sidebar:
    chosen = st.radio("🌍 Dil / Language", list(lang_names.values()),
                       index=list(lang_names.keys()).index(st.session_state.lang))
    st.session_state.lang = [k for k, v in lang_names.items() if v == chosen][0]

lang = st.session_state.lang
L = UI[lang]

with st.sidebar:
    st.markdown(f"### {L['history_title']}")
    sheet = get_sheet()
    sheet_rows = []
    if sheet is not None:
        try:
            sheet_rows = sheet.get_all_records()
        except Exception:
            sheet_rows = []
    if sheet_rows:
        import pandas as pd
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

    st.markdown("---")
    page_labels = L["nav"]
    selected_label = st.radio("Menü", page_labels, index=st.session_state.page, label_visibility="collapsed")
    st.session_state.page = page_labels.index(selected_label)

page = st.session_state.page

def translit(text):
    m = str.maketrans("ğĞşŞıİçÇöÖüÜ", "gGsSiIcCoOuU")
    return text.translate(m)

def build_pdf(image, class_name_local, confidence, description, symptoms, pathogen, spread, treatment, prevention, lang):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    epw = pdf.w - 2 * pdf.l_margin

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_x(pdf.l_margin)
    pdf.cell(epw, 10, "TomatoVision AI - Diagnosis Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin)
    pdf.cell(epw, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
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

    def safe_line(font_style, font_size, text, line_h=7):
        pdf.set_font("Helvetica", font_style, font_size)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(epw, line_h, text)
        pdf.ln(1)

    safe_line("B", 13, translit(f"Prediction: {class_name_local}"), 8)
    safe_line("", 11, f"Confidence: {confidence:.2f}%")
    safe_line("", 11, translit(f"Description: {description}"))
    safe_line("", 11, translit(f"Symptoms: {symptoms}"))
    safe_line("", 11, translit(f"Pathogen: {pathogen}"))
    safe_line("", 11, translit(f"Spread: {spread}"))
    safe_line("", 11, translit(f"Treatment: {treatment}"))
    safe_line("", 11, translit(f"Prevention: {prevention}"))

    return bytes(pdf.output())

# ================= SAYFA 0: ANA SAYFA =================
if page == 0:
    st.markdown(f"""
    <div class="tv-hero">
        <div class="emoji-row">🍅🌿</div>
        <h1>{L['hero_title']}</h1>
        <p>{L['hero_sub']}</p>
    </div>
    """, unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="tv-glass" style="text-align:center;">🧠<br><b>EfficientNetB0</b><br><span style="font-size:0.85rem;color:#8FAE9A;">Deep Learning</span></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="tv-glass" style="text-align:center;">🌍<br><b>3 Dil</b><br><span style="font-size:0.85rem;color:#8FAE9A;">TR / EN / SO</span></div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="tv-glass" style="text-align:center;">📄<br><b>PDF Rapor</b><br><span style="font-size:0.85rem;color:#8FAE9A;">Tek tıkla indir</span></div>', unsafe_allow_html=True)

    st.write("")
    if st.button(L["start_btn"], use_container_width=True, type="primary"):
        st.session_state.page = 1
        st.rerun()

# ================= SAYFA 1: YAPRAK ANALİZİ =================
elif page == 1:
    st.markdown(f'<div class="tv-section-title">{L["nav"][1]}</div>', unsafe_allow_html=True)

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
        is_healthy = predicted_class == "Tomato_healthy"
        class_name_local = info["name"][lang]
        rlevel = risk_level[predicted_class]

        if current_hash != st.session_state.last_hash:
            if sheet is not None:
                try:
                    sheet.append_row([
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        class_name_local,
                        round(confidence, 2)
                    ])
                except Exception:
                    pass
            st.session_state.last_hash = current_hash

        with col2:
            accent = "#4FCB6B" if is_healthy else "#E4573D"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=confidence,
                number={"suffix": "%", "font": {"family": "IBM Plex Mono", "size": 28, "color": "#EAF2EC"}},
                gauge={"axis": {"range": [0, 100], "tickcolor": "#8FAE9A"},
                       "bar": {"color": accent}, "bgcolor": "rgba(255,255,255,0.05)", "borderwidth": 0,
                       "steps": [{"range": [0, 50], "color": "rgba(255,255,255,0.05)"},
                                 {"range": [50, 100], "color": "rgba(255,255,255,0.09)"}]}
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        card_class = "healthy" if is_healthy else "disease"
        icon = "✅" if is_healthy else "⚠️"
        risk_class = {"low": "risk-low", "mid": "risk-mid", "high": "risk-high"}[rlevel]
        risk_text = {"low": L["risk_low"], "mid": L["risk_mid"], "high": L["risk_high"]}[rlevel]

        st.markdown(f"""
        <div class="tv-glass {card_class}">
            <div class="tv-label">{L['diagnosis']}</div>
            <h2>{icon} {class_name_local}</h2>
            <span class="risk-badge {risk_class}">{L['risk']}: {risk_text}</span>
            <div class="tv-label">{L['description']}</div>
            <p>{info['description'][lang]}</p>
            <div class="tv-label">{L['symptoms']}</div>
            <p>{info['symptoms'][lang]}</p>
            <div class="tv-label">{L['pathogen']}</div>
            <p>{info['pathogen'][lang]}</p>
            <div class="tv-label">{L['spread']}</div>
            <p>{info['spread'][lang]}</p>
            <div class="tv-label">{L['treatment'] if not is_healthy else L['care']}</div>
            <p>{info['treatment'][lang] if not is_healthy else info['prevention'][lang]}</p>
            <div class="tv-label">{L['prevention']}</div>
            <p style="margin-bottom:0;">{info['prevention'][lang]}</p>
        </div>
        <p style="font-size:0.82rem;color:#8FAE9A;margin-top:0.6rem;">ℹ️ {L['reliability_note']}</p>
        """, unsafe_allow_html=True)

        if confidence < 50:
            st.warning(L["low_conf"])

        pdf_bytes = build_pdf(image, class_name_local, confidence,
                               info['description'][lang], info['symptoms'][lang], info['pathogen'][lang],
                               info['spread'][lang], info['treatment'][lang], info['prevention'][lang], lang)
        st.download_button(L["download_pdf"], pdf_bytes, "max01_rapor.pdf", "application/pdf")

        # Top 3
        st.markdown(f'<div class="tv-section-title">{L["top3"]}</div>', unsafe_allow_html=True)
        top3_idx = np.argsort(prediction[0])[::-1][:3]
        for idx in top3_idx:
            cname = disease_info[classes[idx]]["name"][lang]
            cprob = float(prediction[0][idx]) * 100
            st.markdown(f"""<div class="top3-row"><span>{cname}</span><span><b>{cprob:.2f}%</b></span></div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="tv-section-title">{L["all_classes"]}</div>', unsafe_allow_html=True)
        sorted_pairs = sorted(zip(classes, prediction[0]), key=lambda x: x[1])
        sorted_labels = [disease_info[c]["name"][lang] for c, _ in sorted_pairs]
        sorted_values = [float(p) * 100 for _, p in sorted_pairs]
        bar_colors = ["#4FCB6B" if c == "Tomato_healthy" else "#E4573D" for c, _ in sorted_pairs]

        fig_bar = go.Figure(go.Bar(
            x=sorted_values, y=sorted_labels, orientation="h", marker_color=bar_colors,
            text=[f"{v:.2f}%" for v in sorted_values], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#EAF2EC")
        ))
        fig_bar.update_layout(
            xaxis=dict(range=[0, 105], title="%", gridcolor="rgba(255,255,255,0.08)", color="#EAF2EC"),
            yaxis=dict(title="", color="#EAF2EC"),
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Work Sans", color="#EAF2EC")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander(L["raw_data"]):
            for cls, prob in sorted(zip(classes, prediction[0]), key=lambda x: -x[1]):
                st.write(f"{disease_info[cls]['name'][lang]}: {prob*100:.2f}%")

# ================= SAYFA 2: HASTALIK BİLGİSİ =================
elif page == 2:
    st.markdown(f'<div class="tv-section-title">{L["disease_db_title"]}</div>', unsafe_allow_html=True)
    st.caption(L["disease_db_sub"])
    options = [disease_info[c]["name"][lang] for c in classes if c != "Tomato_healthy"]
    chosen_name = st.selectbox("", options, label_visibility="collapsed")
    chosen_class = [c for c in classes if disease_info[c]["name"][lang] == chosen_name][0]
    info = disease_info[chosen_class]
    rlevel = risk_level[chosen_class]
    risk_class = {"low": "risk-low", "mid": "risk-mid", "high": "risk-high"}[rlevel]
    risk_text = {"low": L["risk_low"], "mid": L["risk_mid"], "high": L["risk_high"]}[rlevel]

    st.markdown(f"""
    <div class="tv-glass disease">
        <h2>🌿 {info['name'][lang]}</h2>
        <span class="risk-badge {risk_class}">{L['risk']}: {risk_text}</span>
        <div class="tv-label">📖 {L['description']}</div>
        <p>{info['description'][lang]}</p>
        <div class="tv-label">🔍 {L['symptoms']}</div>
        <p>{info['symptoms'][lang]}</p>
        <div class="tv-label">🦠 {L['pathogen']}</div>
        <p>{info['pathogen'][lang]}</p>
        <div class="tv-label">🌡️ {L['spread']}</div>
        <p>{info['spread'][lang]}</p>
        <div class="tv-label">💊 {L['treatment']}</div>
        <p>{info['treatment'][lang]}</p>
        <div class="tv-label">🚜 {L['prevention']}</div>
        <p style="margin-bottom:0;">{info['prevention'][lang]}</p>
    </div>
    """, unsafe_allow_html=True)

# ================= SAYFA 3: TARIM DESTEĞİ =================
elif page == 3:
    st.markdown(f'<div class="tv-section-title">{L["support_title"]}</div>', unsafe_allow_html=True)
    st.caption(L["support_sub"])
    region = st.selectbox(L["support_region_label"], regions[lang])
    st.markdown(f"""
    <div class="tv-glass">
        <p><b>{region}</b></p>
        <p>{L['support_generic']}</p>
        <p>{L['support_line1']}</p>
        <p>{L['support_line2']}</p>
        <p>{L['support_line3']}</p>
    </div>
    <p style="font-size:0.82rem;color:#8FAE9A;margin-top:0.6rem;">ℹ️ {L['support_note']}</p>
    """, unsafe_allow_html=True)

# ================= SAYFA 4: HAKKINDA =================
elif page == 4:
    st.markdown(f"""
    <div class="about-card">
        <div style="font-size:2.5rem;">🍅</div>
        <h2 style="font-family:'Fraunces',serif;">{L['about_title']}</h2>
        <p>{L['about_dev']}</p>
        <p>{L['about_uni']}</p>
        <p>{L['about_dept']}</p>
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:#8FAE9A;">{L['about_stack']}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f'<div class="tv-footer">{L["footer"]}</div>', unsafe_allow_html=True)
