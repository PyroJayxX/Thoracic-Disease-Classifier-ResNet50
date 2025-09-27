import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from keras.models import load_model
import tensorflow as tf
import json

# Load the model (no custom objects needed for binary crossentropy)
model = load_model('./output/baseline_resnet50_best.keras')

class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]

# Your optimized thresholds
optimal_thresholds = {
    "Atelectasis": 0.1,
    "Cardiomegaly": 0.25,
    "Consolidation": 0.1,
    "Edema": 0.30,
    "Effusion": 0.1,
    "Emphysema": 0.15,
    "Fibrosis": 0.25,
    "Hernia": 0.35,
    "Infiltration": 0.15,
    "Mass": 0.15,
    "Nodule": 0.15,
    "Pleural_Thickening": 0.15,
    "Pneumonia": 0.25,
    "Pneumothorax": 0.15
}

def preprocess_image(image_path):
    # Load image in color (medical images are often grayscale but model expects RGB)
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    model_img = cv2.resize(orig_img, (224, 224))
    
    # Normalize pixel range to [0, 1]
    model_img = model_img.astype(np.float32) / 255.0
    
    # Add batch dimension
    model_img = np.expand_dims(model_img, axis=0)  # (1, 224, 224, 3)
    
    return model_img

def generate_gradcam_multilabel(model, img_array, class_index, last_conv_layer_name="conv5_block3_out"):
    """Generate Grad-CAM for a specific class in multi-label setting - using working approach"""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Use specific class index instead of argmax
        class_output = predictions[:, class_index]

    # Compute gradients of top predicted class w.r.t. output feature map
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize between 0 and 1 - same as working code
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def predict_with_thresholds(predictions, thresholds, class_names):
    """Apply optimized thresholds to get binary predictions"""
    detected_diseases = []
    all_results = []
    
    for i, class_name in enumerate(class_names):
        probability = float(predictions[i])
        threshold = thresholds[class_name]
        is_detected = probability >= threshold
        
        result = {
            'class': class_name,
            'probability': probability,
            'threshold': threshold,
            'detected': is_detected
        }
        all_results.append(result)
        
        if is_detected:
            detected_diseases.append(result)
    
    return detected_diseases, all_results

class XrayClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ResNet-50 Multi-label Thoracic Disease Classifier")
        self.setGeometry(200, 200, 1200, 700)  # Wider for multi-label display

        # Create main layout
        main_layout = QHBoxLayout()

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
        main_layout.addLayout(image_layout)

        # Create vertical layout for predictions and buttons
        prediction_layout = QVBoxLayout()

        # Create label for results
        self.results_label = QLabel("")
        self.results_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # Changed alignment for better readability
        self.results_label.setMinimumSize(600, 500)  # Larger area for multi-label results
        self.results_label.setStyleSheet("background-color: #333333; border: 1px solid #444444; padding: 15px; border-radius: 10px; color: white;")
        self.results_label.setWordWrap(True)  # Enable word wrapping
        prediction_layout.addWidget(self.results_label)

        # Create button layout
        button_layout = QHBoxLayout()

        # Create button to classify image
        self.classify_button = QPushButton("Classify Image")
        self.classify_button.setStyleSheet("background-color: #218838; color: white; padding: 10px; border-radius: 5px;")
        self.classify_button.clicked.connect(self.classify_image)
        button_layout.addWidget(self.classify_button)

        prediction_layout.addLayout(button_layout)
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
            self.heatmap_label.setText("Heatmap will be displayed here")
            
    def classify_image(self):
        if not hasattr(self, 'image_path'):
            self.results_label.setText("<h2 style='color: #ff0000;'>No image selected</h2>")
            return

        try:
            # Preprocess image and get predictions
            model_img = preprocess_image(self.image_path)
            predictions = model.predict(model_img, verbose=0)[0]
            
            # Apply optimized thresholds
            detected_diseases, all_results = predict_with_thresholds(predictions, optimal_thresholds, class_names)
            
            # Build result text
            result_text = "<h1 style='color: #00ff00; font-size: 28px; margin-bottom: 15px;'>MULTI-LABEL DIAGNOSIS</h1>"
            
            if detected_diseases:
                result_text += "<h2 style='color: #ffcc00; font-size: 22px; margin-bottom: 10px;'>DETECTED ABNORMALITIES:</h2>"
                for disease in detected_diseases:
                    confidence = disease['probability'] * 100
                    threshold_pct = disease['threshold'] * 100
                    result_text += f"<p style='margin: 5px 0; color: #ff6666; font-size: 16px; font-weight: bold;'>"
                    result_text += f"• {disease['class']}: {confidence:.1f}% (thresh: {threshold_pct:.0f}%)</p>"
                
                # Generate heatmap for the highest confidence detected disease
                highest_confidence_disease = max(detected_diseases, key=lambda x: x['probability'])
                disease_index = class_names.index(highest_confidence_disease['class'])
                
                try:
                    print(f"Generating heatmap for {highest_confidence_disease['class']} (index {disease_index})")
                    heatmap = generate_gradcam_multilabel(model, model_img, disease_index)
                    self.display_heatmap(heatmap, highest_confidence_disease['class'])
                except Exception as e:
                    print(f"Error generating heatmap: {e}")
                    import traceback
                    traceback.print_exc()
                    self.heatmap_label.setText(f"Heatmap generation failed: {str(e)}")
                    
            else:
                result_text += "<h2 style='color: #00ff00; font-size: 20px;'>NO SIGNIFICANT ABNORMALITIES DETECTED</h2>"
                result_text += "<p style='color: #cccccc; font-size: 14px;'>All disease probabilities are below their optimized thresholds.</p>"
                self.heatmap_label.setText("No abnormalities detected - no heatmap generated")

            # Show detailed probabilities
            result_text += "<br><h3 style='color: #00ff00; font-size: 18px; margin-top: 20px;'>DETAILED ANALYSIS:</h3>"
            
            # Sort by probability for better readability
            sorted_results = sorted(all_results, key=lambda x: x['probability'], reverse=True)
            
            for result in sorted_results:
                prob_pct = result['probability'] * 100
                thresh_pct = result['threshold'] * 100
                status = "DETECTED" if result['detected'] else "Not detected"
                color = "#ff6666" if result['detected'] else "#cccccc"
                
                result_text += f"<p style='margin: 2px 0; color: {color}; font-size: 13px;'>"
                result_text += f"{result['class']:<20}: {prob_pct:5.1f}% | {status} (thresh: {thresh_pct:.0f}%)</p>"

            self.results_label.setText(result_text)
            
        except Exception as e:
            error_msg = f"<h2 style='color: #ff0000;'>Error during classification:</h2><p style='color: #ffcccc;'>{str(e)}</p>"
            self.results_label.setText(error_msg)

    def display_heatmap(self, heatmap, disease_name):
        """Display the Grad-CAM heatmap overlaid on the original image - with data type fixes"""
        try:
            print(f"Heatmap dtype: {heatmap.dtype}, shape: {heatmap.shape}")
            
            # Convert to float32 if needed and ensure it's contiguous
            if heatmap.dtype != np.float32:
                heatmap = heatmap.astype(np.float32)
            
            # Ensure array is contiguous
            heatmap = np.ascontiguousarray(heatmap)
            
            # Resize heatmap - try with explicit dtype
            heatmap_resized = cv2.resize(heatmap.astype(np.float32), (400, 400))
            
            # Scale to 0-255 and convert to uint8
            heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
            
            # Apply color map
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # Load and resize original image
            orig_img = cv2.imread(self.image_path)
            orig_img = cv2.resize(orig_img, (400, 400))
            overlay = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)

            # Convert to RGB for Qt display
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            qimage = QImage(overlay_rgb.data, overlay_rgb.shape[1], overlay_rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.heatmap_label.setPixmap(pixmap)
            
            self.heatmap_label.setToolTip(f"Heatmap for: {disease_name}")
            
        except Exception as e:
            print(f"Error in display_heatmap: {e}")
            import traceback
            traceback.print_exc()
            self.heatmap_label.setText(f"Heatmap display failed: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XrayClassifierApp()
    window.show()
    sys.exit(app.exec_())