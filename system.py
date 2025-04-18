import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from tensorflow.keras.models import load_model

model = load_model('thoracic_classifier.h5')

class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
               'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 
               'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']


def preprocess_image(image_path):
    # Load the image
    orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply CLAHE for improved contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(orig_img)
    
    # Create a displayable version of the CLAHE image
    display_img = cv2.resize(clahe_img, (400, 400), interpolation=cv2.INTER_AREA)
    
    # Resize the image to the input size of the model
    model_img = cv2.resize(clahe_img, (224, 224))
    
    # Convert the image to a float32 numpy array and normalize it
    model_img = np.array(model_img, dtype=np.float32) / 255.0
    
    # Expand dimensions to match the input shape of the model (1, 224, 224, 3)
    model_img = np.expand_dims(model_img, axis=0)  # Add batch dimension: (1, 224, 224)
    model_img = np.expand_dims(model_img, axis=3)  # Add channel dimension: (1, 224, 224, 1)
    model_img = np.repeat(model_img, 3, axis=3)    # Repeat to 3 channels: (1, 224, 224, 3)
    
    return model_img, display_img


class XrayClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thoracic Disease Classifier")
        self.setGeometry(200, 200, 800, 600)  # Larger window for better visualization

        # Create layout
        main_layout = QVBoxLayout()
        
        # Create horizontal layout for images
        images_layout = QHBoxLayout()
        
        # Create vertical layout for original image and its label
        original_layout = QVBoxLayout()
        original_label = QLabel("Original Image")
        original_label.setAlignment(Qt.AlignCenter)
        original_label.setFont(QFont('Arial', 12, QFont.Bold))
        original_layout.addWidget(original_label)
        
        # Create label to display original image
        self.original_image_label = QLabel("No image selected")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 400)
        original_layout.addWidget(self.original_image_label)
        
        # Create vertical layout for processed image and its label
        processed_layout = QVBoxLayout()
        processed_label = QLabel("CLAHE Enhancement")
        processed_label.setAlignment(Qt.AlignCenter)
        processed_label.setFont(QFont('Arial', 12, QFont.Bold))
        processed_layout.addWidget(processed_label)
        
        # Create label to display processed image
        self.processed_image_label = QLabel("No processed image")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setMinimumSize(400, 400)
        processed_layout.addWidget(self.processed_image_label)
        
        # Add both layouts to the images layout
        images_layout.addLayout(original_layout)
        images_layout.addLayout(processed_layout)
        
        main_layout.addLayout(images_layout)
        
        # Create label for results
        self.results_label = QLabel("")
        self.results_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.results_label)

        # Create button layout
        button_layout = QHBoxLayout()
        
        # Create button to load image
        self.load_button = QPushButton("Load X-ray Image")
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)

        # Create button to classify image
        self.classify_button = QPushButton("Classify Image")
        self.classify_button.clicked.connect(self.classify_image)
        button_layout.addWidget(self.classify_button)
        
        main_layout.addLayout(button_layout)

        # Set the layout for the main window
        self.setLayout(main_layout)

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
            
            # Process and display CLAHE image
            _, clahe_img = preprocess_image(file_name)
            
            # Convert OpenCV image to Qt format
            h, w = clahe_img.shape
            q_img = QImage(clahe_img.data, w, h, w, QImage.Format_Grayscale8)
            clahe_pixmap = QPixmap.fromImage(q_img)
            
            # Display processed image
            self.processed_image_label.setPixmap(clahe_pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            
            # Clear previous results
            self.results_label.setText("")

    def classify_image(self):
        if hasattr(self, 'image_path'):
            model_img, _ = preprocess_image(self.image_path)
            predictions = model.predict(model_img)[0]
            
            # Get the top 5 predictions
            top_indices = predictions.argsort()[-5:][::-1]
            top_classes = [(class_names[i], predictions[i] * 100) for i in top_indices]
            
            # Format the result text
            result_text = "Predictions:\n"
            for cls, score in top_classes:
                result_text += f"{cls}: {score:.2f}%\n"
            
            self.results_label.setText(result_text)
            self.results_label.adjustSize()
        else:
            self.results_label.setText("No image selected")
            self.results_label.adjustSize()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XrayClassifierApp()
    window.show()
    sys.exit(app.exec_())