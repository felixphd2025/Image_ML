import streamlit as st

import streamlit_classification_app
import model_comp_streamlit
import kernal_tutorial
import test

def main():
    st.title('ML image classification')

    page = st.sidebar.selectbox("Choose a page", [ "Home page", "model training", "kernal tutorial", "Image Classification", "Model Evaluation"])

    if page == "Home page":
        st.header("Home page")
        st.write("This is a simple image classification web app to predict the image class")
        st.write("Navigate to the other pages to use")

    elif page == "model training":
        st.header("model training")
        st.write("set parameters for model training")
        test.main()

    elif page == "kernal tutorial":
        st.header("kernal tutorial")
        st.write("This is a simple image classification web app to predict the image class")
        st.write("Upload an image and the model will predict the class")
        kernal_tutorial.main()

    elif page == "Image Classification":
        st.header("Image Classification")
        st.write("This is a simple image classification web app to predict the image class")
        st.write("Upload an image and the model will predict the class")
        streamlit_classification_app.main()

    elif page == "Model Evaluation":
        st.header("Model Evaluation")
        st.write("This app evaluates the performance of the trained model on the testing dataset.")
        model_comp_streamlit.main()

if __name__ == "__main__":
    main()

