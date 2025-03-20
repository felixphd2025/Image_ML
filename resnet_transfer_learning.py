
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


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
for layer in base_model.layers:
    layer.trainable = False  # Freezing the base model layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(4, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

#### fine tune this model on my image data from mri 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")

# Plot accuracy
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='Train Accuracy')
ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()
plt.show()

# Save model
model.save("model_resnet50.h5")
print("Model saved as model_resnet50.h5")







