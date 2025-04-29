import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from keras.models import load_model
from keras.losses import Loss  
from PyQt5.QtWidgets import QFrame
import os
import tensorflow as tf


custom_objects = {'loss': Loss}  

# def resource_path(relative_path):
#     if hasattr(sys, '_MEIPASS'):
#         return os.path.join(sys._MEIPASS, relative_path)
#     return os.path.join(os.path.abspath("."), relative_path)

# model_path = resource_path("thoracic_classifierV8.keras")
# model = load_model(model_path, custom_objects={'loss': Loss})

model = load_model("./output/thoracic_classifierV8.keras", custom_objects=custom_objects)

class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "No Finding", 
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]


def preprocess_image(image_path):
    # Load grayscale image
    orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to model input size
    model_img = cv2.resize(orig_img, (224, 224))
    
    # Normalize pixel range to [0, 1]
    model_img = model_img.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    model_img = np.expand_dims(model_img, axis=(0, -1))  # (1, 224, 224, 1)
    model_img = np.repeat(model_img, 3, axis=-1)         # (1, 224, 224, 3)
    
    return model_img

import tensorflow as tf

def generate_gradcam(model, img_array, last_conv_layer_name="conv5_block3_out", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    # Compute gradients of top predicted class w.r.t. output feature map
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

class XrayClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thoracic Disease Classifier")
        self.setGeometry(200, 200, 900, 600)  # Adjusted height for heatmap display

        # Create main layout
        main_layout = QHBoxLayout()  # Horizontal layout for image and prediction

        # Create vertical layout for original image and its label
        image_layout = QVBoxLayout()

        # Create label to display original image
        self.original_image_label = QLabel("No image selected")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 400)
        self.original_image_label.setStyleSheet("border: 2px solid #444444; border-radius: 10px; background-color: #333333; color: white;")
        image_layout.addWidget(self.original_image_label)

        # Create label to display heatmap
        self.heatmap_label = QLabel("Heatmap will be displayed here")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumSize(400, 400)
        self.heatmap_label.setStyleSheet("border: 2px solid #444444; border-radius: 10px; background-color: #333333; color: white;")
        image_layout.addWidget(self.heatmap_label)

        button_layout_load = QHBoxLayout()

        # Create button to load image
        self.load_button = QPushButton("Load X-ray Image")
        self.load_button.setStyleSheet("background-color: #005a9e; color: white; padding: 10px; border-radius: 5px;")
        self.load_button.clicked.connect(self.load_image)
        button_layout_load.addWidget(self.load_button)

        image_layout.addLayout(button_layout_load)

        # Add the image layout to the main layout
        main_layout.addLayout(image_layout)

        # Create vertical layout for predictions and buttons
        prediction_layout = QVBoxLayout()

        # Create label for results
        self.results_label = QLabel("")
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setStyleSheet("background-color: #333333; border: 1px solid #444444; padding: 10px; border-radius: 10px; color: white;")
        prediction_layout.addWidget(self.results_label)

        # Create button layout
        button_layout = QHBoxLayout()

        # Create button to classify image
        self.classify_button = QPushButton("Classify Image")
        self.classify_button.setStyleSheet("background-color: #218838; color: white; padding: 10px; border-radius: 5px;")
        self.classify_button.clicked.connect(self.classify_image)
        button_layout.addWidget(self.classify_button)

        prediction_layout.addLayout(button_layout)

        # Add the prediction layout to the main layout
        main_layout.addLayout(prediction_layout)

        # Set the layout for the main window
        self.setLayout(main_layout)
        self.setStyleSheet("background-color: #222222;")

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load X-ray Image", "", 
                                                 "Images (*.png *.jpg *.jpeg);;All Files (*)", 
                                                 options=options)
        if file_name:
            # Display original image
            pixmap = QPixmap(file_name)
            self.original_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            self.image_path = file_name
            
            # Clear previous results
            self.results_label.setText("")
            
    def classify_image(self):
        if hasattr(self, 'image_path'):
            model_img = preprocess_image(self.image_path)
            predictions = model.predict(model_img)[0]
            
            # Get the top prediction
            top_index = np.argmax(predictions)
            top_class = class_names[top_index]
            top_confidence = predictions[top_index] * 100
            
            # Display the top prediction with stylized text
            result_text = f"<h1 style='color: #00ff00; margin=0; font-size: 48px;'>Top Prediction:</h1>"
            result_text += f"<h1 style='color: #ffcc00; margin=0; font-size: 84px;'>{top_class}</h1>"
            result_text += f"<h2 style='color: #00ff00; margin=0; font-size: 36px;'>Confidence: {top_confidence:.2f}%</h2>"

            # Display detailed probabilities for all classes
            result_text += "<h2 style='color: #00ff00; font-size: 22px;'>Detailed Probabilities:</h2>"
            for i, cls in enumerate(class_names):
                prob = predictions[i] * 100
                result_text += f"<p style='margin: 0; color: #ffffff; font-size: 14px;'>{cls}: {prob:.2f}%</p>"

            self.results_label.setText(result_text)
            self.results_label.adjustSize()

            # Generate Grad-CAM heatmap
            heatmap = generate_gradcam(model, model_img)
            heatmap = cv2.resize(heatmap, (400, 400))  # Resize to match QLabel size
            heatmap = np.uint8(255 * heatmap)  # Scale to 0-255
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map

            # Overlay heatmap on the original image
            orig_img = cv2.imread(self.image_path)
            orig_img = cv2.resize(orig_img, (400, 400))
            overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

            # Convert to QPixmap and display in heatmap_label
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            qimage = QImage(overlay.data, overlay.shape[1], overlay.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.heatmap_label.setPixmap(pixmap)
        else:
            self.results_label.setText("<h2 style='color: #ff0000;'>No image selected</h2>")
            self.results_label.adjustSize()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XrayClassifierApp()
    window.show()
    sys.exit(app.exec_())