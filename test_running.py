

import tensorflow as tf
import numpy as np
import cv2 
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("/Users/user/Documents/code/python/image_analysis/model_v1.h5")


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150)) 
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0 

    prediction = model.predict(img_array)

    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    predicted_class = classes[np.argmax(prediction)]

    print(f"🔹 predicted:{predicted_class} (probability: {np.max(prediction):.2f})")

predict_image('/Users/user/Documents/code/python/image_analysis/mri_data/Testing/meningioma/Te-me_0018.jpg')


# make a confusion matrix and auc for the model on teh testing data

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# Directory for testing data
test_dir = '/Users/user/Documents/code/python/image_analysis/mri_data/Testing'
classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Prepare the testing data generator
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Get true labels and predictions
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

# Plot the ROC curve for each class and the overall model
plt.figure(figsize=(10, 8))
for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})")

# Add the micro-average ROC curve
plt.plot(fpr_micro, tpr_micro, label=f"Overall Model (Micro-Average AUC = {roc_auc_micro:.2f})", color='navy', linestyle='--')

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Per Class and Overall)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()