from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

# YOLO modelini yükle
model_path = '/Users/OPERATOR/Desktop/best14.pt'  # Modelin yolu
model = YOLO(model_path)

# Modelden sınıf isimlerini al
class_names = model.names  # Eğer modelinizin .names özelliği yoksa, bir etiket dosyasından yüklemelisiniz.

# Define meal details (name, type, calories, price)
meal_data = {
    "ayran": {"type": "İçecekler", "calories": 60, "price": 33},
    "ayvatatlisi": {"type": "Tatlılar", "calories": 180, "price": 33},
    "baklava": {"type": "Tatlılar", "calories": 250, "price": 33},
    "bezelye": {"type": "Etsiz Ana Yemekler", "calories": 170, "price": 80},
    "borek": {"type": "Yardımcı Yemekler", "calories": 290, "price": 33},
    "bulgurpilavi": {"type": "Yardımcı Yemekler", "calories": 180, "price": 33},
    "cacik": {"type": "Salatalar ve Mezeler", "calories": 90, "price": 33},
    "cigkofte": {"type": "Etsiz Ana Yemekler", "calories": 250, "price": 80},
    "cikolatalitatli": {"type": "Tatlılar", "calories": 300, "price": 33},
    "portakal": {"type": "Meyve", "calories": 47, "price": 33},
    "muz": {"type": "Meyve", "calories": 88, "price": 33},
    "doner": {"type": "Etli Ana Yemekler", "calories": 350, "price": 100},
    "ekmek": {"type": "Ekmek", "calories": 70, "price": 5},
    "eriste": {"type": "Yardımcı Yemekler", "calories": 200, "price": 33},
    "eristecorbasi": {"type": "Çorbalar", "calories": 120, "price": 26},
    "erişte": {"type": "Yardımcı Yemekler", "calories": 200, "price": 33},
    "haslama": {"type": "Etli Ana Yemekler", "calories": 280, "price": 100},
    "havuclumeze": {"type": "Salatalar ve Mezeler", "calories": 150, "price": 33},
    "icetea": {"type": "İçecekler", "calories": 80, "price": 33},
    "iclikofte": {"type": "Etsiz Ana Yemekler", "calories": 200, "price": 80},
    "izgaratavuk": {"type": "Etli Ana Yemekler", "calories": 250, "price": 100},
    "karniyarik": {"type": "Etli Ana Yemekler", "calories": 300, "price": 100},
    "koftepatates": {"type": "Etli Ana Yemekler", "calories": 400, "price": 100},
    "kola": {"type": "İçecekler", "calories": 90, "price": 33},
    "kremalitavuk": {"type": "Etli Ana Yemekler", "calories": 320, "price": 100},
    "kremsantilitatli": {"type": "Tatlılar", "calories": 250, "price": 33},
    "kurufasulye": {"type": "Etli Ana Yemekler", "calories": 240, "price": 100},
    "makarna": {"type": "Yardımcı Yemekler", "calories": 220, "price": 33},
    "manti": {"type": "Etsiz Ana Yemekler", "calories": 400, "price": 80},
    "mercimek": {"type": "Çorbalar", "calories": 150, "price": 26},
    "patatesliet": {"type": "Etli Ana Yemekler", "calories": 300, "price": 100},
    "patateslitavuk": {"type": "Etli Ana Yemekler", "calories": 280, "price": 100},
    "patlicanlimeze": {"type": "Salatalar ve Mezeler", "calories": 150, "price": 33},
    "pirincpilavi": {"type": "Yardımcı Yemekler", "calories": 210, "price": 33},
    "puding": {"type": "Tatlılar", "calories": 150, "price": 33},
    "salata": {"type": "Salatalar ve Mezeler", "calories": 50, "price": 33},
    "sebzelicorba": {"type": "Çorbalar", "calories": 120, "price": 26},
    "sebzeliturlu": {"type": "Etsiz Ana Yemekler", "calories": 180, "price": 80},
    "sehriyecorbasi": {"type": "Çorbalar", "calories": 130, "price": 26},
    "sekerpare": {"type": "Tatlılar", "calories": 250, "price": 33},
    "su": {"type": "İçecekler", "calories": 0, "price": 33},
    "turlu": {"type": "Etsiz Ana Yemekler", "calories": 180, "price": 80},
    "yogurt": {"type": "Yoğurt", "calories": 58, "price": 33}
}

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # İstekten resmi al
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty image file'}), 400

    try:
        # Resmi OpenCV formatında yükle
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Model ile tahmin yap
        results = model.predict(source=img)

        # Tahmin sonuçlarını JSON olarak döndür (sınıf isimleri, güven oranları, tip, kaloriler, fiyat)
        predicted_details = []
        main_dish_count = 0
        other_items = []

        for result in results:
            for box in result.boxes:
                class_index = int(box.cls)
                class_name = class_names[class_index] if class_index < len(class_names) else "Unknown"
                confidence = box.conf[0].item()  # Confidence score of the detection
                
                # Get the meal details (type, calories, price) from the meal_data dictionary
                if class_name in meal_data:
                    meal_info = meal_data[class_name]
                    predicted_details.append({
                        'meal_name': class_name,
                        'meal_type': meal_info["type"],
                        'calories': meal_info["calories"],
                        'price': meal_info["price"],
                        'confidence': confidence  # Add confidence score to the response
                    })

                    # Check if it's a main dish
                    if "Ana Yemek" in meal_info["type"] and class_name not in ["ekmek", "su"]:
                        main_dish_count += 1
                    else:
                        other_items.append(class_name)

        # Check for menu types
        if main_dish_count == 1 and len(other_items) >= 1:
            menu_type = "Menu1"
        elif main_dish_count == 0 and len(other_items) >= 1:
            menu_type = "Menu2"
        elif main_dish_count == 0 and len(other_items) == 0:
            menu_type = "Menu3"
        else:
            menu_type = "Normal Menu"
        
        # Add menu_type to the response
        return jsonify({
            'predictions': predicted_details,
            'menu_type': menu_type
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=52500)
