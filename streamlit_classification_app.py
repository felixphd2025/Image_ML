import streamlit as st
from PIL import Image
import tempfile
import tensorflow as tf
import numpy as np


def main():

    # Streamlit app
    st.title('Image Classification')
    st.write('This is a simple image classification web app to predict the image class')

    # Load the model
    model = tf.keras.models.load_model("/Users/user/Documents/code/python/image_analysis/model_v1.h5")

    # Define the classes
    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# File uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
    # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

    # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_file_path = temp_file.name

    # Preprocess the image
        img = tf.keras.preprocessing.image.load_img(temp_file_path, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image

    # Predict using the model
        prediction = model.predict(img_array)
        predicted_class = classes[np.argmax(prediction)]

    # Display the prediction
        st.write(f"🔹 Predicted: {predicted_class} (Probability: {np.max(prediction):.2f})")

# if uploaded file contains gl it is glioma, if me it is meningioma, if pi it is pituitary, if no it is no tumor
        if 'gl' in uploaded_file.name:
            st.write('🔹 Actual: Glioma')
        elif 'me' in uploaded_file.name:
            st.write('🔹 Actual: Meningioma')
        elif 'pi' in uploaded_file.name:
            st.write('🔹 Actual: Pituitary')
        elif 'no' in uploaded_file.name:
            st.write('🔹 Actual: No Tumor')
        else:
            st.write('🔹 Actual: Unknown')
    if __name__ == "__main__":
        main()



