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

class XrayClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thoracic Disease Classifier")
        self.setGeometry(200, 200, 900, 500)  # Larger window for better visualization

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
            result_text = f"<h2 style='color: #00ff00; margin=0;'>Top Prediction:</h2>"
            result_text += f"<h1 style='color: #ffcc00; margin=0;'>{top_class}</h1>"
            result_text += f"<h3 style='color: #00ff00; margin=0;'>Confidence: {top_confidence:.2f}%</h3>"

            # Display detailed probabilities for all classes
            result_text += "<h2 style='color: #00ff00;'>Detailed Probabilities:</h2>"
            for i, cls in enumerate(class_names):
                prob = predictions[i] * 100
                result_text += f"<p style='margin: 0; color: #ffffff;'>{cls}: {prob:.2f}%</p>"
            
            self.results_label.setText(result_text)
            self.results_label.adjustSize()
        else:
            self.results_label.setText("<h2 style='color: #ff0000;'>No image selected</h2>")
            self.results_label.adjustSize()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XrayClassifierApp()
    window.show()
    sys.exit(app.exec_())