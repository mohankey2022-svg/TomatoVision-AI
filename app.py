"""
TomatoVision-AI - app.py
Domates yaprağı hastalık tespiti için basit bir Flask web uygulaması.

Kullanım:
    python app.py
Sonra tarayıcıdan http://127.0.0.1:5000 adresine gidip bir yaprak fotoğrafı yükleyin.
"""

import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename

# ----------------------------------------------------------------------------
# Ayarlar
# ----------------------------------------------------------------------------
MODEL_PATH = "TomatoVisionAI_EfficientNetB0.keras"
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# kaustubhb999/tomatoleaf veri setindeki sınıf sırası (ImageDataGenerator
# alfabetik sıraya göre class_indices üretir). Eğer kendi eğitiminde
# train_generator.class_indices farklı çıkarsa, aşağıdaki listeyi ona göre güncelle.
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Kullanıcıya gösterilecek daha okunaklı Türkçe isimler (isteğe bağlı)
CLASS_NAMES_TR = {
    "Tomato___Bacterial_spot": "Bakteriyel Leke",
    "Tomato___Early_blight": "Erken Yanıklık",
    "Tomato___Late_blight": "Geç Yanıklık (Mildiyö)",
    "Tomato___Leaf_Mold": "Yaprak Küfü",
    "Tomato___Septoria_leaf_spot": "Septoria Yaprak Lekesi",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Kırmızı Örümcek Akarı",
    "Tomato___Target_Spot": "Hedef Leke Hastalığı",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Sarı Yaprak Kıvırcıklığı Virüsü",
    "Tomato___Tomato_mosaic_virus": "Mozaik Virüsü",
    "Tomato___healthy": "Sağlıklı",
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

# ----------------------------------------------------------------------------
# Modeli yükle (uygulama açılırken bir kez)
# ----------------------------------------------------------------------------
print(f"Model yükleniyor: {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("Model yüklendi.")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(img_path: str) -> np.ndarray:
    """Görüntüyü modele uygun tensöre çevirir."""
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadı."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Sadece png, jpg, jpeg dosyalarına izin verilir."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        img_array = prepare_image(filepath)
        preds = model.predict(img_array)[0]

        top_idx = int(np.argmax(preds))
        top_class = CLASS_NAMES[top_idx]
        confidence = float(preds[top_idx])

        # Tüm sınıflar için olasılıkları da döndür (yüzde olarak)
        all_probs = {
            CLASS_NAMES_TR.get(CLASS_NAMES[i], CLASS_NAMES[i]): round(float(p) * 100, 2)
            for i, p in enumerate(preds)
        }

        result = {
            "class": top_class,
            "class_tr": CLASS_NAMES_TR.get(top_class, top_class),
            "confidence": round(confidence * 100, 2),
            "all_probabilities": dict(
                sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            ),
        }
        return jsonify(result)

    finally:
        # Yüklenen geçici dosyayı temizle
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
