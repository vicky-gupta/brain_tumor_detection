from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import cv2
app = Flask(__name__)

model_path = 'models/models1.h5'
model = load_model(model_path)
#model._make_predict_function()
model.make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')

def final_res(pred_value):
    if (pred_value == 0):
        res_str = "Gloma Tumor"
    elif (pred_value == 1):
        res_str = "Meningioma Tumor"
    elif (pred_value == 2):
        res_str = "No Tumor"
    elif (pred_value == 3):
        res_str = "Pituitary Tumor"
    else:
        res_str = "No valid prediction recieved!!!"
    return res_str

def model_predict(img, model):
    #img = image.load_img(img, target_size = (224,224))
    #x = image.img_to_arr(img)
    #x = np.expand_dims(x,axis=0)
    img_array=cv2.imread(img)
    img_array=cv2.resize(img_array,(224, 224))
    img_array = tf.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_value = np.argmax(preds)
    prediction = final_res(pred_value)
    return prediction

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET' , 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        result = model_predict(file_path, model)
    else:
        result = "None"
    return result
    

if __name__ == "__main__":
    app.run(debug=True)


