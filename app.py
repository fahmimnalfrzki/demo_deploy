import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

interpreter = tf.lite.Interpreter(model_path="fruit_classifier.tflite")
interpreter.allocate_tensors()  # Alokasikan memori untuk model

# Dapatkan informasi input dan output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ['Banana', 'Apple']

st.title('Banana vs Apple Classifier')

uploaded_file = st.file_uploader("Upload gambar buah", type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = classes[int(predictions[0] > 0.5)]

    st.image(img)
    st.write(predicted_class)
