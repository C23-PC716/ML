import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

from flask import Flask, request, render_template

model = keras.models.load_model("Skin_Disease_2.h5")

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

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return render_template('index.html', response="No file selected.")

        try:
            img = Image.open(file)
            img = img.convert('RGB')
            img = img.resize((160, 160))
            
            x = tf.keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict(images, batch_size=32)
            prediction = np.argmax(classes)
            
            return render_template('index.html', 
                                   response="success: the image was predicted to be classified on the " + DISEASE_DICT.get(prediction))
        
        except Exception as e:
            return render_template('index.html', response="error: " + str(e))

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)