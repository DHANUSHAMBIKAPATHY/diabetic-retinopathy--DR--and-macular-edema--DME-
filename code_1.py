import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model

def clahe(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    # Apply CLAHE to the green channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_channel_clahe = clahe.apply(g_channel)
    img = cv2.merge((b_channel, g_channel_clahe, r_channel))
    return img

def preprocessing_for_model(img):
    img = cv2.resize(img, (256, 256))
    img = clahe(img)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return img

def preprocessing_for_display(img):
    img = clahe(img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def predict_image(img):
    img_batch = np.expand_dims(img, axis=0)
    prediction = model.predict(img_batch)
    return prediction

model = load_model('model1.keras')

def main():
    st.title("Diabetic Retinopathy and Macular Edema Detection")
    st.markdown("Developed by: **Dhanush A, Mohammed Arshak**")
    st.markdown("___")
    st.markdown("This application detects diabetic retinopathy (DR) and macular edema (DME) from retina images.")

    uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        
        img = Image.open(uploaded_file)
        img = np.array(img)

        
        preprocessed_img_for_model = preprocessing_for_model(img)
        preprocessed_img_for_display = preprocessing_for_display(img)
        preprocessed_img_for_display_pil = Image.fromarray(preprocessed_img_for_display)
	
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", width=350)
        with col2:
            st.image(preprocessed_img_for_display_pil, caption="Preprocessed Image", width=350)
  

        prediction = predict_image(preprocessed_img_for_model)
        dr = prediction[0]
        dr = np.argmax(dr, axis=1)
        diabetic_retinopathy_prediction = {0:"No Diabetic Retinopathy", 1:"Mild Diabetic Retinopathy", 2:"Moderate Diabetic Retinopathy", 3:"Severe Diabetic Retinopathy", 4:"Proliferative Diabetic Retinopathy"}
        me = prediction[1]
        me = np.argmax(me, axis=1)
        macular_edema_prediction = {0:"No Macular Edema", 1:"Mild Risk of Macular Edema", 2:"High Risk of Macular Edema"}
        st.write(diabetic_retinopathy_prediction[dr[0]])
        st.write(macular_edema_prediction[me[0]])

    else:
        st.write("Please upload an image to analyze.")

if __name__ == "__main__":
    main()