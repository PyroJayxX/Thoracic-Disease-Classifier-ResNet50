import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from keras.models import load_model
from keras.layers import (Dense, GlobalAveragePooling2D, BatchNormalization, 
                         Conv2D, Multiply, Lambda, Concatenate, Dropout, 
                         Add, Activation, Reshape)
from keras import backend as K
import tensorflow as tf
from PIL import Image

# Define custom objects for CBAM model
def define_custom_objects():
    """Define all custom layers used in the CBAM model"""
    def ca_max_lambda(x):
        return K.max(x, axis=[1,2], keepdims=True)
        
    def sa_avg_lambda(x):
        return K.mean(x, axis=-1, keepdims=True)
        
    def sa_max_lambda(x):
        return K.max(x, axis=-1, keepdims=True)
    
    return {
        'ca_max_lambda': ca_max_lambda,
        'sa_avg_lambda': sa_avg_lambda, 
        'sa_max_lambda': sa_max_lambda,
    }

# Load ResNet-50 + CBAM model
custom_objects = define_custom_objects()
model = load_model('./output/stable_resnet_cbam_best.keras', custom_objects=custom_objects)

class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]

def preprocess_image(image_path):
    """Preprocess image for ResNet-50 + CBAM model"""
    # Load image as RGB (ResNet expects RGB)
    pil_image = Image.open(image_path).convert('RGB')
    pil_image = pil_image.resize((224, 224))
    
    # Convert to numpy array and normalize
    image_array = np.array(pil_image) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_array, axis=0)
    
    return image_batch

def extract_cbam_spatial_attention(model, img_array):
    """Extract pure CBAM spatial attention maps directly"""
    try:
        # Get both high-level and top-level CBAM spatial attention outputs
        cbam_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[
                model.get_layer("high_sp_sa_out").output,  # High-level CBAM features
                model.get_layer("top_sp_sa_out").output    # Top-level CBAM features
            ]
        )
        
        high_features, top_features = cbam_model(img_array)
        
        # CBAM spatial attention should preserve spatial dimensions
        # high_features: [1, 14, 14, channels], top_features: [1, 7, 7, channels]
        
        # For visualization, we'll use the high-level features (larger spatial resolution)
        # Take the mean across channels to get spatial attention map
        spatial_attention = tf.reduce_mean(high_features, axis=-1)[0]  # Remove batch dim
        
        # Normalize the attention map
        attention_min = tf.reduce_min(spatial_attention)
        attention_max = tf.reduce_max(spatial_attention)
        
        if attention_max > attention_min:
            spatial_attention = (spatial_attention - attention_min) / (attention_max - attention_min)
        
        print(f"CBAM Spatial Attention Shape: {spatial_attention.shape}")
        print(f"Attention Range: {tf.reduce_min(spatial_attention):.3f} - {tf.reduce_max(spatial_attention):.3f}")
        
        return spatial_attention.numpy()
        
    except Exception as e:
        print(f"Error extracting CBAM spatial attention: {e}")
        return extract_cbam_channel_attention_fallback(model, img_array)

def extract_cbam_channel_attention_fallback(model, img_array):
    """Fallback: Extract channel attention and create spatial representation"""
    try:
        # Get channel attention outputs
        channel_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[
                model.get_layer("high_ch_ca_out").output,  # After channel attention
                model.get_layer("top_ch_ca_out").output
            ]
        )
        
        high_ch, top_ch = channel_model(img_array)
        
        # Use high-level features for better spatial resolution
        # Channel attention modulates each channel, so we can see which channels are important
        channel_importance = tf.reduce_mean(high_ch, axis=(1, 2))[0]  # [channels]
        
        # Create a spatial representation by weighting spatial locations by channel importance
        spatial_weighted = tf.reduce_sum(high_ch[0] * channel_importance, axis=-1)
        
        # Normalize
        spatial_min = tf.reduce_min(spatial_weighted)
        spatial_max = tf.reduce_max(spatial_weighted)
        
        if spatial_max > spatial_min:
            spatial_weighted = (spatial_weighted - spatial_min) / (spatial_max - spatial_min)
        
        print(f"Channel Attention Fallback Shape: {spatial_weighted.shape}")
        
        return spatial_weighted.numpy()
        
    except Exception as e:
        print(f"Error in channel attention fallback: {e}")
        # Last resort: return a uniform attention map
        return np.ones((14, 14)) * 0.5

class XrayCBAMClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ResNet-50 + CBAM Thoracic Disease Classifier")
        self.setGeometry(200, 200, 1200, 700)

        # Create main layout
        main_layout = QHBoxLayout()

        # Create vertical layout for original image and its label
        image_layout = QVBoxLayout()

        # Original image label
        original_title = QLabel("Original X-Ray")
        original_title.setAlignment(Qt.AlignCenter)
        original_title.setStyleSheet("color: white; font-size: 16px; font-weight: bold; padding: 5px;")
        image_layout.addWidget(original_title)

        self.original_image_label = QLabel("No image selected")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 400)
        self.original_image_label.setStyleSheet("border: 2px solid #444444; border-radius: 10px; background-color: #333333; color: white;")
        image_layout.addWidget(self.original_image_label)

        # CBAM attention heatmap label
        heatmap_title = QLabel("CBAM Attention Map")
        heatmap_title.setAlignment(Qt.AlignCenter)
        heatmap_title.setStyleSheet("color: white; font-size: 16px; font-weight: bold; padding: 5px;")
        image_layout.addWidget(heatmap_title)

        self.heatmap_label = QLabel("CBAM attention map will be displayed here")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumSize(400, 400)
        self.heatmap_label.setStyleSheet("border: 2px solid #444444; border-radius: 10px; background-color: #333333; color: white;")
        image_layout.addWidget(self.heatmap_label)

        # Load button
        button_layout_load = QHBoxLayout()
        self.load_button = QPushButton("Load X-ray Image")
        self.load_button.setStyleSheet("background-color: #005a9e; color: white; padding: 10px; border-radius: 5px; font-size: 14px;")
        self.load_button.clicked.connect(self.load_image)
        button_layout_load.addWidget(self.load_button)
        image_layout.addLayout(button_layout_load)

        main_layout.addLayout(image_layout)

        # Create vertical layout for predictions and buttons
        prediction_layout = QVBoxLayout()

        # Model info
        model_info = QLabel("ResNet-50 + CBAM Multi-Label Classifier")
        model_info.setAlignment(Qt.AlignCenter)
        model_info.setStyleSheet("color: #00ff00; font-size: 18px; font-weight: bold; padding: 10px;")
        prediction_layout.addWidget(model_info)

        # Create label for results
        self.results_label = QLabel("Load an X-ray image to see predictions")
        self.results_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.results_label.setWordWrap(True)
        self.results_label.setMinimumSize(400, 500)
        self.results_label.setStyleSheet("background-color: #333333; border: 1px solid #444444; padding: 15px; border-radius: 10px; color: white; font-size: 12px;")
        prediction_layout.addWidget(self.results_label)

        # Create button layout
        button_layout = QHBoxLayout()

        # Create button to classify image
        self.classify_button = QPushButton("Classify X-Ray (Multi-Label)")
        self.classify_button.setStyleSheet("background-color: #218838; color: white; padding: 12px; border-radius: 5px; font-size: 14px;")
        self.classify_button.clicked.connect(self.classify_image)
        button_layout.addWidget(self.classify_button)

        prediction_layout.addLayout(button_layout)
        main_layout.addLayout(prediction_layout)

        self.setLayout(main_layout)
        self.setStyleSheet("background-color: #222222;")

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load X-ray Image", "", 
                                                 "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)", 
                                                 options=options)
        if file_name:
            # Display original image
            pixmap = QPixmap(file_name)
            self.original_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_path = file_name
            
            # Clear previous results
            self.results_label.setText("Image loaded. Click 'Classify X-Ray' to analyze.")
            self.heatmap_label.setText("CBAM attention map will appear after classification")
            
    def classify_image(self):
        if hasattr(self, 'image_path'):
            try:
                # Update UI
                self.classify_button.setText("Analyzing...")
                self.classify_button.setEnabled(False)
                QApplication.processEvents()
                
                # Preprocess and predict
                model_img = preprocess_image(self.image_path)
                predictions = model.predict(model_img, verbose=0)[0]
                
                # Multi-label classification results
                threshold = 0.5
                positive_findings = []
                all_predictions = []
                
                for i, cls in enumerate(class_names):
                    prob = float(predictions[i])
                    all_predictions.append((cls, prob))
                    if prob > threshold:
                        positive_findings.append((cls, prob))
                
                # Sort by probability
                all_predictions.sort(key=lambda x: x[1], reverse=True)
                
                # Create results text
                result_text = "<h2 style='color: #00ff00; margin-bottom: 10px;'>Multi-Label Classification Results</h2>"
                
                if positive_findings:
                    result_text += "<h3 style='color: #ff6b6b; margin-bottom: 8px;'>🔴 POSITIVE FINDINGS (>50%):</h3>"
                    positive_findings.sort(key=lambda x: x[1], reverse=True)
                    for cls, prob in positive_findings:
                        result_text += f"<p style='margin: 2px 0; color: #ff6b6b; font-weight: bold;'>• {cls}: {prob:.1%}</p>"
                else:
                    result_text += "<h3 style='color: #4ecdc4; margin-bottom: 8px;'>✅ NO SIGNIFICANT FINDINGS</h3>"
                
                result_text += "<h3 style='color: #ffd93d; margin: 15px 0 8px 0;'>All Probabilities:</h3>"
                
                for cls, prob in all_predictions:
                    color = "#ff6b6b" if prob > threshold else "#ffffff"
                    result_text += f"<p style='margin: 1px 0; color: {color};'>{cls:20s}: {prob:6.1%}</p>"
                
                self.results_label.setText(result_text)
                
                # Generate pure CBAM attention map
                try:
                    attention_map = extract_cbam_spatial_attention(model, model_img)
                    
                    # Ensure attention map is valid
                    if attention_map is not None and attention_map.size > 0:
                        # Resize attention map to image size with proper interpolation
                        attention_resized = cv2.resize(attention_map.astype(np.float32), (400, 400), interpolation=cv2.INTER_CUBIC)
                        
                        # Ensure values are in [0, 1] range
                        attention_resized = np.clip(attention_resized, 0, 1)
                        
                        # Convert to uint8 for colormap
                        attention_uint8 = (attention_resized * 255).astype(np.uint8)
                        
                        # Apply colormap
                        heatmap_colored = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)
                        
                        # Load and resize original image for overlay
                        orig_img = cv2.imread(self.image_path)
                        if orig_img is not None:
                            orig_img = cv2.resize(orig_img, (400, 400))
                            
                            # Create overlay with proper data types
                            heatmap_colored = heatmap_colored.astype(np.float32)
                            orig_img = orig_img.astype(np.float32)
                            
                            overlay = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
                            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                            
                            # Convert BGR to RGB for Qt
                            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                            
                            # Create QImage with explicit parameters
                            height, width, channel = overlay_rgb.shape
                            bytes_per_line = 3 * width
                            qimage = QImage(overlay_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                            
                            pixmap = QPixmap.fromImage(qimage)
                            self.heatmap_label.setPixmap(pixmap)
                        else:
                            self.heatmap_label.setText("Error loading original image for overlay")
                    else:
                        self.heatmap_label.setText("Invalid attention map generated")
                        
                except Exception as e:
                    self.heatmap_label.setText(f"Error generating attention map: {str(e)}")
                    print(f"Detailed error: {e}")  # Debug print
                
            except Exception as e:
                self.results_label.setText(f"<h2 style='color: #ff0000;'>Classification Error:</h2><p style='color: white;'>{str(e)}</p>")
                
            finally:
                self.classify_button.setText("Classify X-Ray (Multi-Label)")
                self.classify_button.setEnabled(True)
        else:
            self.results_label.setText("<h2 style='color: #ff0000;'>No image selected</h2>")

def main():
    app = QApplication(sys.argv)
    
    # Check if model exists
    import os
    if not os.path.exists('./output/stable_resnet_cbam_best.keras'):
        print("Error: Model file './output/stable_resnet_cbam_best.keras' not found!")
        print("Make sure your trained model is in the correct location.")
        sys.exit(1)
    
    try:
        window = XrayCBAMClassifierApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Make sure you have PyQt5 installed: pip install PyQt5")

if __name__ == "__main__":
    main()