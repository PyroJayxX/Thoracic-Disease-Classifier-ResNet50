import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, 
                             QPushButton, QFileDialog, QHBoxLayout, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from keras.models import load_model
from keras.losses import Loss
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import tensorflow as tf

custom_objects = {'loss': Loss, 'Addons>SigmoidFocalCrossEntropy': SigmoidFocalCrossEntropy()}
model = load_model('./output/baseline_model_resnet50.keras', custom_objects=custom_objects)

class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]


def preprocess_image(image_path):
    orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    model_img = cv2.resize(orig_img, (224, 224))
    model_img = model_img.astype(np.float32) / 255.0
    model_img = np.expand_dims(model_img, axis=(0, -1))
    model_img = np.repeat(model_img, 3, axis=-1)
    return model_img

def generate_gradcam(model, img_array, last_conv_layer_name="conv5_block3_out", pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


class XrayClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thoracic Disease Classifier")
        self.setGeometry(150, 150, 1200, 700)  # Wider window
        self.setStyleSheet("background-color: #1e1e1e; color: white; font-family: Arial;")

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(30)

        # ----------------- Left: Original Image + Predictions -----------------
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)

        # Original Image
        self.original_frame = QFrame()
        self.original_frame.setStyleSheet("border: 2px solid #555555; border-radius: 10px;")
        self.original_frame.setFixedSize(300, 300)
        self.original_image_label = QLabel("Original Image")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setStyleSheet("background-color: #2e2e2e; border-radius: 10px;")
        layout_orig = QVBoxLayout()
        layout_orig.addWidget(self.original_image_label)
        self.original_frame.setLayout(layout_orig)
        left_layout.addWidget(self.original_frame)

        # Predictions (all visible, no scroll)
        self.results_label = QLabel("Predictions will appear here")
        self.results_label.setAlignment(Qt.AlignTop)
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("""
            background-color: #2e2e2e; 
            border: 2px solid #555555; 
            border-radius: 10px; 
            padding: 10px;
            font-size: 14px;
        """)
        left_layout.addWidget(self.results_label)

        # Load Button
        self.load_button = QPushButton("Load X-ray Image")
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #007acc; 
                color: white; 
                padding: 12px; 
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover { background-color: #005f99; }
        """)
        self.load_button.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_button)

        # Classify Button
        self.classify_button = QPushButton("Classify Image")
        self.classify_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745; 
                color: white; 
                padding: 12px; 
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover { background-color: #218838; }
        """)
        self.classify_button.clicked.connect(self.classify_image)
        left_layout.addWidget(self.classify_button)

        main_layout.addLayout(left_layout)

        # ----------------- Right: Heatmap -----------------
        right_layout = QVBoxLayout()

        self.heatmap_frame = QFrame()
        self.heatmap_frame.setStyleSheet("border: 2px solid #555555; border-radius: 10px;")
        self.heatmap_frame.setFixedSize(700, 700)  # Large heatmap
        self.heatmap_label = QLabel("Grad-CAM Heatmap")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setStyleSheet("background-color: #2e2e2e; border-radius: 10px;")
        layout_heat = QVBoxLayout()
        layout_heat.addWidget(self.heatmap_label)
        self.heatmap_frame.setLayout(layout_heat)
        right_layout.addWidget(self.heatmap_frame)

        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load X-ray Image", "", 
                                                 "Images (*.png *.jpg *.jpeg);;All Files (*)", 
                                                 options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.original_image_label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            self.image_path = file_name
            self.results_label.setText("")

    def classify_image(self):
        if hasattr(self, 'image_path'):
            model_img = preprocess_image(self.image_path)
            predictions = model.predict(model_img)[0]

            top_index = np.argmax(predictions)
            top_class = class_names[top_index]
            top_confidence = predictions[top_index] * 100

            result_text = f"<b style='color:#00ff00;'>Top Prediction:</b> <span style='color:#ffcc00;'>{top_class}</span><br>"
            result_text += f"<b style='color:#00ff00;'>Confidence:</b> {top_confidence:.2f}%<br><br>"
            result_text += "<b style='color:#00ff00;'>All Probabilities:</b><br>"
            for i, cls in enumerate(class_names):
                prob = predictions[i] * 100
                result_text += f"{cls}: {prob:.2f}%<br>"

            self.results_label.setText(result_text)

            # Generate Grad-CAM
            heatmap = generate_gradcam(model, model_img)
            heatmap = cv2.resize(heatmap, (700, 700))  # Match frame size
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            orig_img = cv2.imread(self.image_path)
            orig_img = cv2.resize(orig_img, (700, 700))
            overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            qimage = QImage(overlay.data, overlay.shape[1], overlay.shape[0], QImage.Format_RGB888)
            self.heatmap_label.setPixmap(QPixmap.fromImage(qimage))
        else:
            self.results_label.setText("<b style='color: #ff0000;'>No image selected</b>")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XrayClassifierApp()
    window.show()
    sys.exit(app.exec_())
