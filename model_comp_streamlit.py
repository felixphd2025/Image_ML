import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

def main():
    # Streamlit app
    st.title("Model Evaluation: ROC Curve and Confusion Matrix")
    st.write("This app evaluates the performance of the trained model on the testing dataset.")

    # Load the model
    model = tf.keras.models.load_model("/Users/user/Documents/code/python/image_analysis/model_v1.h5")

    # Directory for testing data
    test_dir = '/Users/user/Documents/code/python/image_analysis/mri_data/Testing'
    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    # Prepare the testing data generator
    st.write("Loading testing data...")
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Get true labels and predictions
    st.write("Generating predictions...")
    y_true = test_generator.classes
    y_true_binarized = label_binarize(y_true, classes=range(len(classes)))

    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Generate ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calculate micro-average ROC curve and AUC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_binarized.ravel(), y_pred.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # Plot the ROC curve
    st.write("### ROC Curve")
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(len(classes)):
        ax.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})")
    ax.plot(fpr_micro, tpr_micro, label=f"Overall Model (Micro-Average AUC = {roc_auc_micro:.2f})", color='navy', linestyle='--')
    ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Per Class and Overall)")
    ax.legend(loc="lower right")
    ax.grid()
    st.pyplot(fig)

    # Generate the confusion matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred_classes)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

if __name__ == "__main__":
    main()