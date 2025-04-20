import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from keras.models import load_model
from keras.losses import Loss  

custom_objects = {'loss': Loss}  # Replace 'loss' with your custom loss function name
model = load_model('thoracic_classifierV3.h5', custom_objects=custom_objects)

class_names = ["Pneumonia", "Cardiomegaly", "Edema", 
                "Emphysema", "Effusion", "Infiltration", "Atelectasis"]


def preprocess_image(image_path):
    # Load grayscale image
    orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to model input size
    model_img = cv2.resize(orig_img, (224, 224))
    
    # Normalize pixel range to [0, 1]
    model_img = model_img.astype(np.float32) / 255.0
    
    # Samplewise centering (subtract mean of this image)
    model_img -= np.mean(model_img)
    
    # Samplewise std normalization (divide by std of this image)
    model_img /= (np.std(model_img) + 1e-7)
    
    # Add batch and channel dimensions
    model_img = np.expand_dims(model_img, axis=(0, -1))  # (1, 224, 224, 1)
    model_img = np.repeat(model_img, 3, axis=-1)         # (1, 224, 224, 3)
    
    return model_img



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
        
        # Add the original image layout to the images layout
        images_layout.addLayout(original_layout)
        
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
            
            # Clear previous results
            self.results_label.setText("")

    def classify_image(self):
        if hasattr(self, 'image_path'):
            model_img = preprocess_image(self.image_path)
            predictions = model.predict(model_img)[0]
            
            # Get the top prediction
            top_index = np.argmax(predictions)
            top_class = class_names[top_index]
            confidence = predictions[top_index] * 100
            
            # Display the top prediction
            result_text = f"Prediction:\n{top_class}\nConfidence: {confidence:.2f}%\n\n"
            
            # Display detailed probabilities for all classes
            result_text += "Detailed Probabilities:\n"
            for i, cls in enumerate(class_names):
                prob = predictions[i] * 100
                result_text += f"{cls}: {prob:.2f}%\n"
            
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