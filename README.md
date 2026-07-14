# 🍅 max01 — TomatoVision AI

Derin öğrenme (EfficientNetB0 tabanlı CNN) kullanarak domates yaprağı fotoğraflarından hastalık tespiti yapan bir web uygulaması.

🔗 **Canlı demo:** https://tomatovision-ai-m6zay2no9tvnq82fxkksbd.streamlit.app

---

## 📋 Proje Hakkında

max01, domates bitkilerinde yaygın görülen 9 farklı hastalığı ve sağlıklı yaprakları, tek bir fotoğraftan otomatik olarak sınıflandıran bir yapay zeka uygulamasıdır. Model, [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) veri setinin domates alt kümesi üzerinde eğitilmiştir.

### Tespit Edilen Sınıflar

| # | Hastalık (TR) | Disease (EN) |
|---|---|---|
| 1 | Bakteriyel Leke Hastalığı | Bacterial Spot |
| 2 | Erken Yanıklık | Early Blight |
| 3 | Geç Yanıklık | Late Blight |
| 4 | Yaprak Küfü | Leaf Mold |
| 5 | Septoria Yaprak Lekesi | Septoria Leaf Spot |
| 6 | Kırmızı Örümcek | Two-Spotted Spider Mite |
| 7 | Hedef Lekesi | Target Spot |
| 8 | Sarı Yaprak Kıvırcıklığı Virüsü | Yellow Leaf Curl Virus |
| 9 | Mozaik Virüsü | Mosaic Virus |
| 10 | Sağlıklı Yaprak | Healthy |

---

## ✨ Özellikler

- 🔍 **Görsel yükleme veya kamera ile** anlık fotoğraf çekme
- 🧠 **EfficientNetB0 tabanlı CNN** ile 150×150 çözünürlükte sınıflandırma
- 🌿 **İlaç / ürün önerisi** ve sağlıklı yapraklar için bakım tavsiyeleri
- 🌍 **Türkçe / İngilizce** dil desteği
- 📄 **PDF rapor** olarak sonuç indirme
- 📊 **Tahmin geçmişi** ve CSV dışa aktarma (oturum bazlı)
- 📱 Mobil uyumlu, özel tasarımlı arayüz
- 📈 Güven skoru göstergesi (gauge grafik) ve tüm sınıf olasılıkları için karşılaştırmalı bar chart

---

## 🛠️ Kullanılan Teknolojiler

- **Model:** TensorFlow / Keras (EfficientNetB0 mimarisi)
- **Arayüz:** Streamlit
- **Görselleştirme:** Plotly
- **PDF Oluşturma:** fpdf2
- **Deploy:** Streamlit Community Cloud
- **Model depolama:** Git LFS (228 MB)

---

## 🚀 Yerel Kurulum

```bash
git clone https://github.com/mohankey2022-svg/TomatoVision-AI.git
cd TomatoVision-AI
pip install -r requirements.txt
streamlit run app.py
```

> Not: Model dosyası (`TomatoVisionAI_EfficientNetB0.keras`) Git LFS ile takip edilmektedir. Klonlama sırasında `git lfs pull` gerekebilir.

---

## 📁 Proje Yapısı
