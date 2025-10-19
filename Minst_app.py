import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

# Load pre-trained MNIST model (or train your model beforehand)
# For demonstration, we will load pre-trained weights if saved
# model = tf.keras.models.load_model("mnist_model.h5")

# Load MNIST test set
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

st.title("MNIST Handwritten Digit Classifier 🖐️")

# Upload image for prediction
uploaded_file = st.file_uploader("Upload a 28x28 grayscale image of a digit", type=["png", "jpg"])

if uploaded_file is not None:
    from PIL import Image

    image = Image.open(uploaded_file).convert('L')  # grayscale
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.image(image, caption="Uploaded Image", width=150)
    st.write(f"Predicted digit: **{predicted_digit}**")
