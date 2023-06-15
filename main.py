import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io, json
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

from flask import Flask, request, Response, render_template
from flask_api import status

model_tf = keras.models.load_model("dermacare.h5")
# model_tflite = keras.models.load_model("bla.tflite")

DISEASE_DICT = {
    0: "Actinic Keratosis",
    1: "Basal Cell Carcinoma",
    2: "Benign keratosis",
    3: "Dermatofibroma",
    4: "Melanocytic Nevus",
    5: "Melanoma",
    6: "Squamous Cell Carcinoma",
    7: "Vascular Lesion",
}

DISEASE_DICT_FINAL = {
    0: {
        "nama": "Actinic Keratosis",
        "deskripsi": "Actinic keratosis adalah kondisi prakanker yang mempengaruhi kulit. Ini disebabkan oleh kerusakan sinar matahari pada kulit selama periode waktu yang panjang.",
    },
    1: {
        "nama" : "Basal Cell Carcinoma",
        "deskripsi" : "Salah satu jenis kanker kulit yang paling umum. Kanker ini terjadi ketika sel-sel basal, yang bertanggung jawab untuk pembentukan kulit baru, mengalami mutasi dan berkembang secara tidak terkendali.",
    },
    2: {
        "nama" : "Benign keratosis",
        "deskripsi" : "Kondisi kulit yang umum terjadi pada orang dewasa yang lebih tua, ini bukanlah kanker kulit, tetapi merupakan pertumbuhan kulit yang tidak berbahaya.",
    },
    3: {
        "nama" : "Dermatofibroma",
        "deskripsi" : "Lesi kulit yang umum, non-kanker, dan umumnya tidak berbahaya. Lesi ini terbentuk ketika sel-sel fibrosit dan sel-sel histiosit mengumpul di dalam dermis, lapisan tengah kulit.",
    },
    4: {
        "nama" : "Melanocytic Nevus",
        "deskripsi" : "Dikenal sebagai tahi lalat atau nevus pigmentosus, adalah lesi kulit yang umum dan biasanya non-kanker. Ini terbentuk ketika sel-sel melanosit, yang bertanggung jawab untuk produksi pigmen melanin, mengumpul di kulit.",
    },
    5: {
        "nama" : "Melanoma",
        "deskripsi" : "Salah satu jenis kanker kulit yang berasal dari sel-sel melanosit, yaitu sel-sel yang menghasilkan pigmen melanin yang memberikan warna pada kulit.",
    },
    6: {
        "nama" : "Squamous Cell Carcinoma",
        "deskripsi" : "Salah satu jenis kanker kulit yang berasal dari sel skuamosa, yaitu sel-sel yang membentuk lapisan tengah dan luar kulit.",
    },
    7: {
        "nama" : "Vascular Lesion",
        "deskripsi" : "Adalah salah satu kelainan atau gangguan pada pembuluh darah atau pembuluh limfe di dalam kulit atau jaringan lainnya.",
    },
}

app = Flask(__name__)

def get_prediction(model, input_file) :
    img = Image.open(input_file)
    img = img.convert('RGB')
    img = img.resize((160, 160))

    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=32)
    prediction_class_index = np.argmax(classes)

    return prediction_class_index

@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return render_template('index.html', response="No file selected.")

        try:
            # prediction = get_prediction(model_tflite, file)
            prediction = get_prediction(model_tf, file)
            return render_template('index.html', 
                                   response="success: the image was predicted to be classified on the " + DISEASE_DICT.get(prediction))
        
        except Exception as e:
            return render_template('index.html', response="error: " + str(e))

    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict_disease():
    
    file = request.files["image"]
    
    if file is None or file.filename == "":
        return Response(
            response= json.dumps({"message" : "No file uploaded"}), 
            status=status.HTTP_400_BAD_REQUEST,
            content_type="application/json")

    elif not (file.filename.endswith(".png") or file.filename.endswith(".jpg") or file.filename.endswith(".jpeg")) :
        return Response(
            response= json.dumps({"message" : "The file is not an image. Try to upload a png, jpg, or a jpeg file!"}), 
            status=status.HTTP_400_BAD_REQUEST,
            content_type="application/json")

    try:
        # prediction_key = get_prediction(model_tflite, file)
        prediction_key = get_prediction(model_tf, file)
        # prediction = DISEASE_DICT.get(prediction_key)
        prediction = DISEASE_DICT_FINAL.get(prediction_key)

        # return Response(
        #     response= json.dumps({"prediction" : prediction}), 
        #     status=status.HTTP_200_OK,
        #     content_type="application/json")

        return Response(
            response= json.dumps(prediction), #return dictionary {nama, deskripsi}
            status=status.HTTP_200_OK,
            content_type="application/json")
        
    except Exception as e:
        return Response(
            response= json.dumps({"message" : str(e)}), 
            status=status.HTTP_400_BAD_REQUEST,
            content_type="application/json")



if __name__ == "__main__":
    app.run(debug=True)
