import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model_path = 'railway_track_fault_detection_cnn_model.h5'  
loaded_model = load_model(model_path)
img_width, img_height = 150, 150

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    prediction = loaded_model.predict(img_array)

    return prediction[0][0]

def main():
    st.title("Railway Track Fault Detection")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as temp_image_file:
            temp_image_file.write(uploaded_file.getvalue())

        result = classify_image(temp_image_path)

        if result < 0.5:
            st.success("Defective")
        else:
            st.success("Non-defective")
        os.remove(temp_image_path)

if __name__ == "__main__":
    main()
