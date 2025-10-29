"""
Hybrid ResNet-50 + CBAM + ViT Thoracic Disease Classifier
Multi-label classification with CBAM attention visualization
"""

import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from PyQt5.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, 
                             QPushButton, QFileDialog, QHBoxLayout, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from tensorflow.keras import layers

# ============================================================================
# CBAM ATTENTION MODULE
# ============================================================================

class ChannelAttention(layers.Layer):
    """Channel Attention Module"""
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_dense_one = layers.Dense(
            channels // self.ratio,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True
        )
        self.shared_dense_two = layers.Dense(
            channels,
            kernel_initializer='he_normal',
            use_bias=True
        )
        super(ChannelAttention, self).build(input_shape)
    
    def call(self, inputs):
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        avg_pool = layers.Reshape((1, 1, inputs.shape[-1]))(avg_pool)
        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)
        
        max_pool = layers.GlobalMaxPooling2D()(inputs)
        max_pool = layers.Reshape((1, 1, inputs.shape[-1]))(max_pool)
        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)
        
        channel_attention = layers.Add()([avg_pool, max_pool])
        channel_attention = layers.Activation('sigmoid')(channel_attention)
        
        return layers.Multiply()([inputs, channel_attention])
    
    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config


class SpatialAttention(layers.Layer):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False
        )
        super(SpatialAttention, self).build(input_shape)
    
    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_attention = self.conv(concat)
        return layers.Multiply()([inputs, spatial_attention])
    
    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class CBAM(layers.Layer):
    """Convolutional Block Attention Module"""
    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_attention = ChannelAttention(ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
    
    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'ratio': self.ratio,
            'kernel_size': self.kernel_size
        })
        return config
    


# ============================================================================
# LOAD MODEL
# ============================================================================

print("Loading hybrid model...")
model = keras.models.load_model("./output/hybrid_model_final.keras", custom_objects={
        'CBAM': CBAM,
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention
    })
print("✓ Model loaded successfully")

# Disease labels
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image_path):
    """Preprocess image for model input"""
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # Preprocess for ResNet
    img_array = img.astype(np.float32)
    img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img

# ============================================================================
# CBAM ATTENTION VISUALIZATION
# ============================================================================

def get_cbam_attention(model, img_array):
    """
    Extract CBAM attention map from the hybrid model
    
    CBAM provides spatial attention which shows where the model
    is focusing for disease localization.
    """
    # Create a model that outputs CBAM layer
    cbam_layer_name = 'cbam'  # Name from your model definition
    
    try:
        cbam_model = keras.models.Model(
            inputs=model.input,
            outputs=model.get_layer(cbam_layer_name).output
        )
        
        # Get CBAM output
        cbam_output = cbam_model.predict(img_array, verbose=0)
        
        # Average across channels to get spatial attention
        attention_map = np.mean(cbam_output[0], axis=-1)
        
        # Normalize to 0-1
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        return attention_map
    
    except Exception as e:
        print(f"Error extracting CBAM attention: {e}")
        # Fallback: use Grad-CAM
        return get_gradcam_fallback(model, img_array)

def get_gradcam_fallback(model, img_array):
    """Fallback to Grad-CAM if CBAM extraction fails"""
    # Find last conv layer in ResNet
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer = layer.name
            break
    
    if last_conv_layer is None:
        return np.ones((7, 7))  # Return blank if no conv layer found
    
    grad_model = keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Use max prediction for visualization
        class_output = tf.reduce_max(predictions)
    
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(original_img, attention_map):
    """Overlay attention heatmap on original image"""
    # Resize attention map to image size
    heatmap = cv2.resize(attention_map, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay

# ============================================================================
# GUI APPLICATION
# ============================================================================

class HybridClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid ResNet-CBAM-ViT Thoracic Disease Classifier")
        self.setGeometry(100, 100, 1200, 700)
        self.image_path = None
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left side: Images
        left_layout = QVBoxLayout()
        
        # Original image
        self.original_label = QLabel("Load a chest X-ray image to begin")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(500, 250)
        self.original_label.setStyleSheet(
            "border: 2px solid #2c3e50; "
            "border-radius: 10px; "
            "background-color: #34495e; "
            "color: #ecf0f1; "
            "font-size: 14px;"
        )
        left_layout.addWidget(QLabel("<h3 style='color: #ecf0f1;'>Original Image</h3>"))
        left_layout.addWidget(self.original_label)
        
        # Attention map
        self.heatmap_label = QLabel("Attention map will appear here")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumSize(500, 250)
        self.heatmap_label.setStyleSheet(
            "border: 2px solid #2c3e50; "
            "border-radius: 10px; "
            "background-color: #34495e; "
            "color: #ecf0f1; "
            "font-size: 14px;"
        )
        left_layout.addWidget(QLabel("<h3 style='color: #ecf0f1;'>Heatmap Activation</h3>"))
        left_layout.addWidget(self.heatmap_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Load X-ray")
        self.load_button.setStyleSheet(
            "background-color: #3498db; "
            "color: white; "
            "padding: 12px; "
            "border-radius: 8px; "
            "font-size: 14px; "
            "font-weight: bold;"
        )
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)
        
        self.classify_button = QPushButton("Prediction")
        self.classify_button.setStyleSheet(
            "background-color: #27ae60; "
            "color: white; "
            "padding: 12px; "
            "border-radius: 8px; "
            "font-size: 14px; "
            "font-weight: bold;"
        )
        self.classify_button.clicked.connect(self.classify_image)
        self.classify_button.setEnabled(False)
        button_layout.addWidget(self.classify_button)
        
        left_layout.addLayout(button_layout)
        
        main_layout.addLayout(left_layout)
        
        # Right side: Predictions
        right_layout = QVBoxLayout()
        
        right_layout.addWidget(QLabel("<h2 style='color: #ecf0f1;'>Predictions</h2>"))
        
        # Scrollable results area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setStyleSheet(
            "background-color: #2c3e50; "
            "border: 2px solid #34495e; "
            "border-radius: 10px;"
        )
        
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_widget.setLayout(self.results_layout)
        scroll.setWidget(self.results_widget)
        
        right_layout.addWidget(scroll)
        
        main_layout.addLayout(right_layout)
        
        self.setLayout(main_layout)
        self.setStyleSheet("background-color: #1a252f;")
    
    def load_image(self):
        """Load X-ray image"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Chest X-ray", 
            "", 
            "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )
        
        if file_name:
            self.image_path = file_name
            
            # Display original
            pixmap = QPixmap(file_name)
            self.original_label.setPixmap(
                pixmap.scaled(500, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            
            # Clear previous results
            self.clear_results()
            self.heatmap_label.setText("Attention map will appear after classification")
            
            # Enable classify button
            self.classify_button.setEnabled(True)
    
    def clear_results(self):
        """Clear previous predictions"""
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            # For non-widget items (like stretch), just remove them
    
    def classify_image(self):
        """Classify image and show results"""
        if not self.image_path:
            return
        
        try:
            # Preprocess
            img_array, original_img = preprocess_image(self.image_path)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)[0]
            
            # Get CBAM attention
            attention_map = get_cbam_attention(model, img_array)
            
            # Overlay and display
            overlay = overlay_heatmap(original_img, attention_map)
            
            # Convert to QPixmap
            h, w, ch = overlay.shape
            bytes_per_line = ch * w
            qimage = QImage(overlay.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.heatmap_label.setPixmap(
                pixmap.scaled(500, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            
            # Clear results
            self.clear_results()
            
            # Exclude 'No Finding' from display
            disease_classes = [class_name for class_name in CLASS_NAMES if class_name != 'No Finding']
            disease_predictions = [(class_name, predictions[i]) for i, class_name in enumerate(CLASS_NAMES) if class_name != 'No Finding']
            disease_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Check if any disease > 50% (excluding 'No Finding')
            has_significant_findings = any(confidence > 0.5 for _, confidence in disease_predictions)
            
            if not has_significant_findings:
                # Show "No significant findings" at the top
                no_disease_label = QLabel(
                    "<div style='padding: 20px;'>"
                    "<h3 style='color: #2ecc71;'>✓ No significant findings</h3>"
                    "<p style='color: #ecf0f1;'>All predictions below 50% confidence</p>"
                    "</div>"
                )
                no_disease_label.setStyleSheet(
                    "background-color: #27ae60; "
                    "border-radius: 8px;"
                )
                self.results_layout.addWidget(no_disease_label)
            
            # Display all disease predictions (excluding 'No Finding')
            for disease, confidence in disease_predictions:
                # Color based on confidence
                if confidence > 0.7:
                    color = "#e74c3c"  # Red for high confidence
                    border_color = "#e74c3c"
                elif confidence > 0.5:
                    color = "#f39c12"  # Orange for medium
                    border_color = "#f39c12"
                else:
                    color = "#2ecc71"  # Green for low
                    border_color = "#34495e"
                
                disease_label = QLabel(
                    f"<div style='padding: 10px; margin: 5px;'>"
                    f"<h3 style='color: {color}; margin: 0;'>{disease}</h3>"
                    f"<p style='color: #ecf0f1; font-size: 16px; margin: 5px 0;'>"
                    f"Confidence: <b>{confidence*100:.1f}%</b></p>"
                    f"</div>"
                )
                disease_label.setStyleSheet(
                    f"background-color: #34495e; "
                    "border-radius: 8px; "
                    f"border: 2px solid {border_color};"
                )
                self.results_layout.addWidget(disease_label)
            
            # Add spacer
            self.results_layout.addStretch()
            
        except Exception as e:
            error_label = QLabel(
                f"<div style='padding: 20px;'>"
                f"<h3 style='color: #e74c3c;'>Error</h3>"
                f"<p style='color: #ecf0f1;'>{str(e)}</p>"
                f"</div>"
            )
            self.clear_results()
            self.results_layout.addWidget(error_label)
            
        except Exception as e:
            error_label = QLabel(
                f"<div style='padding: 20px;'>"
                f"<h3 style='color: #e74c3c;'>Error</h3>"
                f"<p style='color: #ecf0f1;'>{str(e)}</p>"
                f"</div>"
            )
            self.clear_results()
            self.results_layout.addWidget(error_label)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HybridClassifierApp()
    window.show()
    sys.exit(app.exec_())