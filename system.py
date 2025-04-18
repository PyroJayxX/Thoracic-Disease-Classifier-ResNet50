import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model

model = load_model('thoracic_classifier.h5')

class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
               'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 
               'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']


def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Resize the image to the input size of the model
    img = cv2.resize(img, (224, 224))
    # Convert the image to a float32 numpy array and normalize it
    img = np.array(img, dtype=np.float32) / 255.0
    # Expand dimensions to match the input shape of the model (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)  # Add batch dimension: (1, 224, 224)
    img = np.expand_dims(img, axis=3)  # Add channel dimension: (1, 224, 224, 1)
    img = np.repeat(img, 3, axis=3)    # Repeat to 3 channels: (1, 224, 224, 3)
    return img

class XrayClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thoracic Disease Classifier")
        self.setGeometry(200, 200, 400, 400)

        # Create layout
        layout = QVBoxLayout()

        # Create label to display image
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # Create button to load image
        self.load_button = QPushButton("Load X-ray Image")
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        # Create button to classify image
        self.classify_button = QPushButton("Classify Image")
        self.classify_button.clicked.connect(self.classify_image)
        layout.addWidget(self.classify_button)

        # Set the layout for the main window
        self.setLayout(layout)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load X-ray Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            self.image_path = file_name

    def classify_image(self):
        if hasattr(self, 'image_path'):
            img = preprocess_image(self.image_path)
            predictions = model.predict(img)
            predicted_class = class_names[np.argmax(predictions)]
            confidence_score = np.max(predictions) * 100

            result_text = f"Predicted Class: {predicted_class}\nConfidence Score: {confidence_score:.2f}%"
            self.image_label.setText(result_text)
            self.image_label.adjustSize()
        else:
            self.image_label.setText("No image selected")
            self.image_label.adjustSize()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XrayClassifierApp()
    window.show()
    sys.exit(app.exec_())
# This code is a simple PyQt5 application that allows users to load an X-ray image and classify it using a pre-trained model.