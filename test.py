
import os
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

DATASET_PATH = "/Users/user/Documents/code/python/image_analysis/mri_data" 

# this takes the images and augments them to create more data for the model to train on
train_datagen = ImageDataGenerator(
    #rescales data by bringing images into the range of 0-1.
    rescale=1./255, 
    # roatates the image within a range of 20 degrees randomly
    rotation_range=20, 
    # randomly shifts the image horizontall by 20% of the total width
    width_shift_range=0.2, 
    # randomly shifts the image vertically by 20% of the total height
    height_shift_range=0.2, 
    # randomly shears the image by 20%
    shear_range=0.2, 
    # randomly zooms the image in or out by 20%
    zoom_range=0.2, 
    # randomly flips the image horizontally
    horizontal_flip=True, 
    # reserves 10% of the data for validation
    validation_split=0.1  # 10%
)

# Train data generator. This will load and augment training images.
train_generator = train_datagen.flow_from_directory(
    # Specifies the path to the "Training" folder in the dataset.
    os.path.join(DATASET_PATH, "Training"),
    # Resizes all images to 150x150 pixels to standardize the input size for the model.
    target_size=(150, 150),  
    # Specifies that 32 images will be processed in each batch.
    batch_size=32,  
    # Defines the type of labels to use. 'categorical' means the labels are one-hot encoded.
    class_mode='categorical',
    # Indicates that this generator should use the subset of data meant for training (90% of the data).
    subset='training'  
)

# Validation data generator. This will load and augment validation images.
val_generator = train_datagen.flow_from_directory(
    # Specifies the path to the "Training" folder in the dataset, which contains both training and validation data.
    os.path.join(DATASET_PATH, "Training"),
    # Resizes all images to 150x150 pixels to standardize the input size for the model.
    target_size=(150, 150),
    # Specifies that 32 images will be processed in each batch.
    batch_size=32,
    # Defines the type of labels to use. 'categorical' means the labels are one-hot encoded.
    class_mode='categorical',   
    # Indicates that this generator should use the subset of data meant for validation (10% of the data).
    subset='validation'  
)
# Test data generator. This will load and preprocess testing images (without augmentation).
test_datagen = ImageDataGenerator(rescale=1./255)
# Test generator for loading and preprocessing the test images.
test_generator = test_datagen.flow_from_directory(
    # Specifies the path to the "Testing" folder in the dataset.
    os.path.join(DATASET_PATH, "Testing"),
    # Resizes all images to 150x150 pixels to standardize the input size for the model.
    target_size=(150, 150),
    # Specifies that 32 images will be processed in each batch.
    batch_size=32,
    # Defines the type of labels to use. 'categorical' means the labels are one-hot encoded.
    class_mode='categorical'
)

def main():
    st.title("MRI Image Classification")

    # Input activation function for model
    activation_function = st.selectbox("Select activation function", ["relu", "tanh", "sigmoid", "softmax"])
    activation_flatten = st.selectbox("Select activation function for flatten layer", ["relu", "tanh", "sigmoid", "softmax"])
    epochs_number = st.number_input("Number of epochs", min_value=1, max_value=100, value=10, step=1)

    if st.button("Train Model"):
        with st.spinner("Training the model..."):
            model = Sequential([
                Conv2D(32, (3, 3), activation=activation_function, input_shape=(150, 150, 3)),
                MaxPooling2D(2, 2),

                Conv2D(64, (3, 3), activation=activation_function),
                MaxPooling2D(2, 2),

                Conv2D(128, (3, 3), activation=activation_function),
                MaxPooling2D(2, 2),

                Flatten(),
                Dense(512, activation=activation_function),
                Dropout(0.5),
                Dense(4, activation=activation_flatten)  # 4 (Glioma, Meningioma, Pituitary, No Tumor)
            ])

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            st.text("Model Summary:")
            model.summary(print_fn=lambda x: st.text(x))

            history = model.fit(
                train_generator,
                epochs=epochs_number,
                validation_data=val_generator
            )

            test_loss, test_acc = model.evaluate(test_generator)
            st.success(f"Test Accuracy: {test_acc:.2f}")

            # Plot accuracy
            st.subheader("Training vs Validation Accuracy")
            fig, ax = plt.subplots()
            ax.plot(history.history['accuracy'], label='Train Accuracy')
            ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            st.pyplot(fig)

            # Save model
            model.save("model_v1.h5")
            st.success("Model saved as model_v1.h5")

if __name__ == "__main__":
    main()





