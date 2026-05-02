import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from keras.models import load_model
from keras.losses import Loss
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from PyQt5.QtWidgets import QFrame
import tensorflow as tf
from tensorflow.keras import layers

# =============================================================================
# CUSTOM LAYERS (Same as in training notebook)
# =============================================================================

class PatchExtractor(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) should be divisible by num_heads ({num_heads})")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads
        })
        return config

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout
        
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        mlp_output = self.mlp(out1)
        mlp_output = self.dropout2(mlp_output, training=training)
        return self.layernorm2(out1 + mlp_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout_rate
        })
        return config

class CrossAttentionFusion(layers.Layer):
    def __init__(self, attention_dim, num_heads=8, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        
        # Project CNN and ViT features to common dimension
        self.cnn_projection = layers.Dense(attention_dim)
        self.vit_projection = layers.Dense(attention_dim)
        
        # Cross-attention layer
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=attention_dim // num_heads,
            dropout=dropout
        )
        
        # Layer normalization and dropout
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        cnn_features, vit_features = inputs
        
        # Project features to common dimension
        cnn_proj = self.cnn_projection(cnn_features)
        vit_proj = self.vit_projection(vit_features)
        
        # Cross-attention: CNN features as queries, ViT features as keys/values
        attended_features = self.cross_attention(
            query=cnn_proj,
            key=vit_proj,
            value=vit_proj,
            training=training
        )
        
        # Residual connection and normalization
        output = self.layernorm(cnn_proj + self.dropout(attended_features, training=training))
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "attention_dim": self.attention_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate
        })
        return config

# =============================================================================
# MODEL LOADING AND CONFIGURATION
# =============================================================================

# Custom objects for loading the model
custom_objects = {
    'loss': Loss,
    'Addons>SigmoidFocalCrossEntropy': SigmoidFocalCrossEntropy(),
    'PatchExtractor': PatchExtractor,
    'PatchEncoder': PatchEncoder,
    'MultiHeadSelfAttention': MultiHeadSelfAttention,
    'TransformerBlock': TransformerBlock,
    'CrossAttentionFusion': CrossAttentionFusion
}

# Load the trained model
print("Loading ResNet50-ViT Cross-Attention model...")
try:
    model = load_model('./output/resnet50_vit_best_model.keras', custom_objects=custom_objects)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model file exists and was trained with the same architecture.")
    sys.exit(1)

# Class names (including No_Finding)
class_names = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "No_Finding", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# =============================================================================
# IMAGE PREPROCESSING AND GRAD-CAM
# =============================================================================

def preprocess_image(image_path):
    """Preprocess image for the ResNet50-ViT model"""
    # Load grayscale image
    orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to model input size
    model_img = cv2.resize(orig_img, (224, 224))
    
    # Convert to RGB (repeat grayscale channel)
    model_img = cv2.cvtColor(model_img, cv2.COLOR_GRAY2RGB)
    
    # Normalize pixel range to [0, 1]
    model_img = model_img.astype(np.float32) / 255.0
    
    # Add batch dimension
    model_img = np.expand_dims(model_img, axis=0)  # (1, 224, 224, 3)
    
    return model_img

def generate_gradcam_resnet_vit(model, img_array, layer_name="resnet50", pred_index=None):
    """Generate Grad-CAM for ResNet50-ViT hybrid model"""
    try:
        # Create a model that maps the input image to the activations of the last conv layer
        # and the output predictions
        last_conv_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )

        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # The gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # Vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        # Return a dummy heatmap if Grad-CAM fails
        return np.zeros((7, 7))

def generate_attention_visualization(model, img_array):
    """Generate attention visualization from the cross-attention layer"""
    try:
        # Get the cross-attention layer
        cross_attention_layer = None
        for layer in model.layers:
            if isinstance(layer, CrossAttentionFusion):
                cross_attention_layer = layer
                break
        
        if cross_attention_layer is None:
            print("Cross-attention layer not found, using Grad-CAM instead")
            return generate_gradcam_resnet_vit(model, img_array)
        
        # Create a model that outputs attention weights
        # This is a simplified version - you might need to modify based on your exact architecture
        intermediate_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=model.get_layer('predictions').output
        )
        
        # For now, fall back to Grad-CAM
        return generate_gradcam_resnet_vit(model, img_array)
        
    except Exception as e:
        print(f"Error generating attention visualization: {e}")
        return generate_gradcam_resnet_vit(model, img_array)

# =============================================================================
# GUI APPLICATION
# =============================================================================

class XrayClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ResNet50-ViT Cross-Attention Thoracic Disease Classifier")
        self.setGeometry(200, 200, 1000, 700)  # Wider for better display

        # Create main layout
        main_layout = QHBoxLayout()

        # Create vertical layout for original image and its label
        image_layout = QVBoxLayout()

        # Create label to display original image
        self.original_image_label = QLabel("No image selected")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 400)
        self.original_image_label.setStyleSheet(
            "border: 2px solid #444444; border-radius: 10px; "
            "background-color: #333333; color: white; font-size: 14px;"
        )
        image_layout.addWidget(self.original_image_label)

        # Create label to display heatmap/attention visualization
        self.heatmap_label = QLabel("Attention visualization will be displayed here")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumSize(400, 400)
        self.heatmap_label.setStyleSheet(
            "border: 2px solid #444444; border-radius: 10px; "
            "background-color: #333333; color: white; font-size: 14px;"
        )
        image_layout.addWidget(self.heatmap_label)

        # Load image button
        button_layout_load = QHBoxLayout()
        self.load_button = QPushButton("Load X-ray Image")
        self.load_button.setStyleSheet(
            "background-color: #005a9e; color: white; padding: 10px; "
            "border-radius: 5px; font-size: 14px; font-weight: bold;"
        )
        self.load_button.clicked.connect(self.load_image)
        button_layout_load.addWidget(self.load_button)
        image_layout.addLayout(button_layout_load)

        # Add the image layout to the main layout
        main_layout.addLayout(image_layout)

        # Create vertical layout for predictions and buttons
        prediction_layout = QVBoxLayout()

        # Model info label
        model_info = QLabel("ResNet50-ViT Cross-Attention Fusion Model")
        model_info.setAlignment(Qt.AlignCenter)
        model_info.setStyleSheet(
            "background-color: #1a1a1a; color: #00ff00; padding: 10px; "
            "border-radius: 5px; font-size: 16px; font-weight: bold;"
        )
        prediction_layout.addWidget(model_info)

        # Create label for results
        self.results_label = QLabel("Load an X-ray image to get predictions")
        self.results_label.setAlignment(Qt.AlignTop)
        self.results_label.setMinimumHeight(500)
        self.results_label.setStyleSheet(
            "background-color: #333333; border: 1px solid #444444; "
            "padding: 15px; border-radius: 10px; color: white; font-size: 12px;"
        )
        prediction_layout.addWidget(self.results_label)

        # Create button layout
        button_layout = QHBoxLayout()

        # Create button to classify image
        self.classify_button = QPushButton("Analyze Image")
        self.classify_button.setStyleSheet(
            "background-color: #218838; color: white; padding: 10px; "
            "border-radius: 5px; font-size: 14px; font-weight: bold;"
        )
        self.classify_button.clicked.connect(self.classify_image)
        self.classify_button.setEnabled(False)  # Disabled until image is loaded
        button_layout.addWidget(self.classify_button)

        # Clear button
        self.clear_button = QPushButton("Clear Results")
        self.clear_button.setStyleSheet(
            "background-color: #dc3545; color: white; padding: 10px; "
            "border-radius: 5px; font-size: 14px; font-weight: bold;"
        )
        self.clear_button.clicked.connect(self.clear_results)
        button_layout.addWidget(self.clear_button)

        prediction_layout.addLayout(button_layout)

        # Add the prediction layout to the main layout
        main_layout.addLayout(prediction_layout)

        # Set the layout for the main window
        self.setLayout(main_layout)
        self.setStyleSheet("background-color: #222222;")

    def load_image(self):
        """Load and display X-ray image"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load X-ray Image", "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)", 
            options=options
        )
        if file_name:
            # Display original image
            pixmap = QPixmap(file_name)
            self.original_image_label.setPixmap(
                pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self.image_path = file_name
            
            # Enable classify button
            self.classify_button.setEnabled(True)
            
            # Clear previous results
            self.results_label.setText("Image loaded. Click 'Analyze Image' to get predictions.")
            self.heatmap_label.setText("Attention visualization will appear here after analysis.")
            
    def classify_image(self):
        """Classify the loaded X-ray image"""
        if hasattr(self, 'image_path'):
            try:
                # Preprocess image
                model_img = preprocess_image(self.image_path)
                
                # Get predictions
                predictions = model.predict(model_img, verbose=0)[0]
                
                # Find top predictions
                top_indices = np.argsort(predictions)[::-1][:5]  # Top 5 predictions
                
                # Create results display
                result_text = "<h2 style='color: #00ff00; margin: 0; text-align: center;'>Analysis Results</h2><br>"
                
                # Top prediction
                top_class = class_names[top_indices[0]]
                top_confidence = predictions[top_indices[0]] * 100
                result_text += f"<div style='background-color: #1a4d3a; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                result_text += f"<h3 style='color: #00ff88; margin: 0;'>Primary Finding:</h3>"
                result_text += f"<h2 style='color: #ffcc00; margin: 5px 0;'>{top_class}</h2>"
                result_text += f"<p style='color: #88ff88; margin: 0; font-size: 16px;'>Confidence: {top_confidence:.1f}%</p>"
                result_text += "</div><br>"

                # Top 5 predictions
                result_text += "<h3 style='color: #00ff00; margin: 10px 0;'>Top 5 Findings:</h3>"
                for i, idx in enumerate(top_indices):
                    cls = class_names[idx]
                    prob = predictions[idx] * 100
                    
                    # Color coding based on confidence
                    if prob > 50:
                        color = "#ff4444"  # High confidence - red
                    elif prob > 20:
                        color = "#ffaa44"  # Medium confidence - orange
                    else:
                        color = "#ffffff"  # Low confidence - white
                    
                    result_text += f"<p style='margin: 3px 0; color: {color}; font-size: 14px;'>"
                    result_text += f"{i+1}. {cls}: {prob:.1f}%</p>"

                # All class probabilities (sorted)
                result_text += "<br><h4 style='color: #888888; margin: 10px 0;'>All Classes (sorted by confidence):</h4>"
                sorted_indices = np.argsort(predictions)[::-1]
                for idx in sorted_indices:
                    cls = class_names[idx]
                    prob = predictions[idx] * 100
                    result_text += f"<p style='margin: 1px 0; color: #cccccc; font-size: 11px;'>{cls}: {prob:.1f}%</p>"

                self.results_label.setText(result_text)

                # Generate and display attention visualization
                try:
                    heatmap = generate_attention_visualization(model, model_img)
                    
                    # Resize heatmap to match display size
                    heatmap_resized = cv2.resize(heatmap, (400, 400))
                    
                    # Convert to 8-bit and apply colormap
                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                    # Load and resize original image for overlay
                    orig_img = cv2.imread(self.image_path)
                    orig_img = cv2.resize(orig_img, (400, 400))
                    
                    # Create overlay
                    overlay = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)

                    # Convert to QPixmap and display
                    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    qimage = QImage(
                        overlay_rgb.data, 
                        overlay_rgb.shape[1], 
                        overlay_rgb.shape[0], 
                        QImage.Format_RGB888
                    )
                    pixmap = QPixmap.fromImage(qimage)
                    self.heatmap_label.setPixmap(pixmap)
                    
                except Exception as e:
                    print(f"Error generating attention visualization: {e}")
                    self.heatmap_label.setText(f"Attention visualization failed: {str(e)}")

            except Exception as e:
                error_msg = f"<h3 style='color: #ff0000;'>Error during analysis:</h3><p style='color: #ffaa00;'>{str(e)}</p>"
                self.results_label.setText(error_msg)
                print(f"Classification error: {e}")
        else:
            self.results_label.setText("<h3 style='color: #ff0000;'>No image loaded</h3>")

    def clear_results(self):
        """Clear all results and reset the interface"""
        self.original_image_label.setText("No image selected")
        self.original_image_label.setPixmap(QPixmap())
        self.heatmap_label.setText("Attention visualization will be displayed here")
        self.heatmap_label.setPixmap(QPixmap())
        self.results_label.setText("Load an X-ray image to get predictions")
        self.classify_button.setEnabled(False)
        if hasattr(self, 'image_path'):
            delattr(self, 'image_path')

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = XrayClassifierApp()
    window.show()
    
    print("ResNet50-ViT Cross-Attention Classifier App started!")
    print("Model loaded with", len(class_names), "classes:", class_names)
    
    sys.exit(app.exec_())