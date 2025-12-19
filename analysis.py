import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

MODEL_PATH = r"D:\Project\gesture_mobilenet_best.keras"
TEST_DIR = r"D:\Project\dataset\test"

IMG_SIZE = 160
BATCH_SIZE = 32


model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded")


test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_data.class_indices.keys())
print("Classes:", class_names)


test_loss, test_acc = model.evaluate(test_data, verbose=1)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_data.classes


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
