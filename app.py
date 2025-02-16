import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

model = load_model('fruit_classifier.h5')
classes = ['Banana', 'Apple']

st.title('Banana vs Apple Classifier')

uploaded_file = st.file_uploader("Upload gambar buah", type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = classes[int(predictions[0] > 0.5)]
    confidence = predictions[0][0]

    st.image(img, caption=f'Prediksi: {predicted_class} ({confidence:.2f})', use_column_width=True)
