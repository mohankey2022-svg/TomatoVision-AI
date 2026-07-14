# 🍅 TomatoVision-AI

Domates yaprağı fotoğraflarından hastalık tespiti yapan, **EfficientNetB0** tabanlı transfer learning modeli ve bu modeli servis eden basit bir Flask web uygulaması.

## Proje Yapısı

```
TomatoVision-AI/
│── app.py                              # Flask web uygulaması (çıkarım / inference)
│── templates/
│   └── index.html                      # Yükleme arayüzü
│── TomatoVisionAI_EfficientNetB0.keras  # Eğitilmiş model dosyası
│── requirements.txt                    # Python bağımlılıkları
│── README.md
```

## Model Hakkında

Model, [`kaustubhb999/tomatoleaf`](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf) veri seti üzerinde eğitilmiştir ve aşağıdaki 10 sınıfı ayırt eder:

| # | Sınıf |
|---|-------|
| 1 | Tomato___Bacterial_spot |
| 2 | Tomato___Early_blight |
| 3 | Tomato___Late_blight |
| 4 | Tomato___Leaf_Mold |
| 5 | Tomato___Septoria_leaf_spot |
| 6 | Tomato___Spider_mites Two-spotted_spider_mite |
| 7 | Tomato___Target_Spot |
| 8 | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
| 9 | Tomato___Tomato_mosaic_virus |
| 10 | Tomato___healthy |

**Mimari:** `EfficientNetB0` (ImageNet ağırlıkları, `include_top=False`) + `GlobalAveragePooling2D` + `Dense(256, relu)` + `Dropout(0.5)` + `Dense(10, softmax)`.

**Eğitim stratejisi:**
1. Base model dondurulmuş halde (`trainable=False`) 10 epoch eğitim.
2. Ardından son 20 katman açılarak (`fine-tuning`) düşük öğrenme oranıyla (`1e-5`) 5 epoch daha eğitim.

> ⚠️ **Önemli:** Bu repodaki `.keras` dosyasının, notebook'taki eğitim tamamlandıktan sonra kaydedilmesi gerekir. Notebook'un sonuna aşağıdaki satırı ekleyip çalıştırman yeterli:
> ```python
> model.save("TomatoVisionAI_EfficientNetB0.keras")
> ```
> Bu dosyayı ürettikten sonra proje klasörüne kopyala.

## Kurulum

```bash
git clone <repo-url>
cd TomatoVision-AI
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
python app.py
```

Tarayıcıdan `http://127.0.0.1:5000` adresine gidip bir domates yaprağı fotoğrafı yükle. Model, en olası hastalığı ve tüm sınıflar için olasılık dağılımını gösterecektir.

## API Kullanımı

Uygulama, arayüz dışında doğrudan `curl` ile de kullanılabilir:

```bash
curl -X POST -F "file=@yaprak.jpg" http://127.0.0.1:5000/predict
```

Örnek yanıt:

```json
{
  "class": "Tomato___Late_blight",
  "class_tr": "Geç Yanıklık (Mildiyö)",
  "confidence": 92.31,
  "all_probabilities": {
    "Geç Yanıklık (Mildiyö)": 92.31,
    "Erken Yanıklık": 4.12,
    "...": "..."
  }
}
```

## Notlar

- Görseller modele verilmeden önce `224x224` boyutuna yeniden boyutlandırılır ve `1/255` ile normalize edilir (eğitimdeki `ImageDataGenerator` ayarlarıyla tutarlı).
- `CLASS_NAMES` listesinin sırası, `ImageDataGenerator.flow_from_directory` çağrısının ürettiği alfabetik `class_indices` sırasına göre ayarlanmıştır. Kendi eğitiminde `train_generator.class_indices` çıktısı farklıysa `app.py` içindeki listeyi buna göre güncelle.
- Üretim (production) ortamında `debug=True` ayarını kapat ve `flask run` yerine `gunicorn` gibi bir WSGI sunucusu kullan.

## Lisans

Bu proje eğitim/araştırma amaçlıdır.
