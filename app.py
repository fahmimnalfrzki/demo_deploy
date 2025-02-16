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

# Buat dummy input sesuai dengan format yang diharapkan model
input_shape = input_details[0]['shape']  # Misalnya [1, 224, 224, 3] untuk gambar
dummy_input = np.random.rand(*input_shape).astype(np.float32)  # Simulasi input

# Set input ke model
interpreter.set_tensor(input_details[0]['index'], dummy_input)

# Jalankan inferensi
interpreter.invoke()

# Ambil hasil output
predictions = interpreter.get_tensor(output_details[0]['index'])

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
