# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 17:42:26 2025

@author: GITAA029
"""

import streamlit as st
import numpy as np
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
import cv2

model = keras.models.load_model("digit_model.h5")

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) in the box below")

canvas = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas.image_data is not None:
        img = canvas.image_data[:, :, 0]
        img = cv2.resize(img, (28, 28))
        img = 255 - img
        img = img / 255.0
        img = img.reshape(1, 784)

        prediction = model.predict(img)
        digit = np.argmax(prediction)

        st.success(f"🎯 Predicted Digit: **{digit}**")
