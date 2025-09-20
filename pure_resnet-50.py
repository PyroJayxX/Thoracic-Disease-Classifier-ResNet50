import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import glob


BASE_DIR = "./dataset_balanced"
CSV_PATH = os.path.join(BASE_DIR, "Data_Entry_2017.csv")

# Updated image directory path to match your structure
IMAGE_DIR = os.path.join(BASE_DIR, "new_images", "new_images")
IMAGE_FILES = {os.path.basename(f): f for f in glob.glob(os.path.join(IMAGE_DIR, "*.png"))}

# Check if images were found
print(f"Total images found: {len(IMAGE_FILES)}")
print(f"Looking for images in: {IMAGE_DIR}")

# Verify the directory exists
if not os.path.exists(IMAGE_DIR):
    print(f"ERROR: Image directory does not exist: {IMAGE_DIR}")
    print("Please check your directory structure.")
else:
    print("Image directory found successfully!")

# Load CSV
df = pd.read_csv(CSV_PATH)

# Keep only relevant columns
df = df[['Image Index', 'Finding Labels']]

# Filter to only include images that actually exist in our dataset
df = df[df['Image Index'].isin(IMAGE_FILES.keys())]
print(f"Images found in CSV that exist in directory: {len(df)}")

# Convert multi-labels into lists
df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))

# Copy to avoid modifying original
df_filtered = df.copy()

# Binarize the labels
mlb = MultiLabelBinarizer()
binarized_labels = mlb.fit_transform(df_filtered['Finding Labels'])

# Convert binarized labels into a DataFrame with column names
label_columns = mlb.classes_  # Get class names for column headers
df_labels = pd.DataFrame(binarized_labels, columns=label_columns)

# Concatenate with the original DataFrame
df_filtered = pd.concat([df_filtered[['Image Index']], df_labels], axis=1)

# Check label distribution
print("Label distribution:")
print(df_labels.sum(axis=0))  # Sum of each class occurrence

print("\nFirst few rows of filtered data:")
print(df_filtered.head())

valid_labels = mlb.classes_
print(f"\nValid labels ({len(valid_labels)}):")
print(valid_labels)  # This is your ordered list of label names

print("\nSample of Finding Labels with Pneumonia:")
print(df[df['Finding Labels'].apply(lambda x: 'Pneumonia' in x)].head(10))

print("\nMost common label combinations:")
print(df['Finding Labels'].apply(lambda x: ','.join(x)).value_counts().head(10))

# Prepare input (X) and labels (y)
X = df_filtered['Image Index'].values  # Image filenames
y = df_filtered[valid_labels].values   # Multi-label one-hot encoding

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

print(f"\nDataset split:")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Augmentation settings for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# No augmentation for validation/test (only rescale)
val_test_datagen = ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# Train generator with augmentation - FIXED PATH
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_filtered[df_filtered['Image Index'].isin(X_train)],
    directory=IMAGE_DIR,  # Updated to use the correct path
    x_col="Image Index",
    y_col=valid_labels,  # Multi-label columns
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

# Validation generator (no augmentation) - FIXED PATH
val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=df_filtered[df_filtered['Image Index'].isin(X_val)],
    directory=IMAGE_DIR,  # Updated to use the correct path
    x_col="Image Index",
    y_col=valid_labels,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

# Test generator (for final evaluation) - FIXED PATH
test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=df_filtered[df_filtered['Image Index'].isin(X_test)],
    directory=IMAGE_DIR,  # Updated to use the correct path
    x_col="Image Index",
    y_col=valid_labels,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False
)

print(f"\nData generators created:")
print(f"Train batches: {len(train_generator)}")
print(f"Validation batches: {len(val_generator)}")
print(f"Test batches: {len(test_generator)}")

# Load ResNet-50 with pretrained weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False  # Prevents modification

# Add new classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x_class = Dense(128, activation='relu')(x)
x_class = Dropout(0.3)(x_class)
output_class = Dense(len(valid_labels), activation='sigmoid', name='classification_output')(x_class)

# Define and compile model
model = Model(inputs=base_model.input, outputs=output_class)
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=1e-4),
              metrics=['accuracy', 'AUC'])

print(f"\nModel created with {len(valid_labels)} output classes")
print("Model summary:")
print(model.summary())

# Save the best model based on validation AUC
checkpoint = ModelCheckpoint('classification_head.h5', monitor='val_auc', mode='max', save_best_only=True, verbose=1)

# Reduce LR if validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)

callbacks = [checkpoint, reduce_lr]

print("\n" + "="*50)
print("PHASE 1: Training classification head only")
print("="*50)

# Train the classification head only
history = model.fit(
    train_generator,
    epochs=5,  
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks
)

print("\n" + "="*50)
print("PHASE 2: Fine-tuning with unfrozen layers")
print("="*50)

# Unfreeze last 10 layers of ResNet-50
for layer in base_model.layers[-10:]:
    layer.trainable = True

finetune_checkpoint = ModelCheckpoint('thoracic_classifier.h5', monitor='val_auc', mode='max', save_best_only=True, verbose=1)
finetune_callbacks = [finetune_checkpoint, reduce_lr]

# Recompile with a smaller learning rate for fine-tuning
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=1e-5),  # Lower LR
              metrics=['accuracy', 'AUC'])

# Train again with fine-tuning
history_finetune = model.fit(
    train_generator,
    epochs=5, 
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=finetune_callbacks
)

print("\n" + "="*50)
print("FINAL EVALUATION")
print("="*50)

# Load the best model for evaluation
best_model = load_model('thoracic_classifier.h5')

# Make predictions on test set
print("Making predictions on test set...")
test_predictions = best_model.predict(test_generator)

# Get true labels for test set
test_true_labels = []
test_generator.reset()  # Reset generator to start from beginning
for i in range(len(test_generator)):
    _, labels = next(test_generator)
    test_true_labels.extend(labels)

test_true_labels = np.array(test_true_labels)

print(f"Test predictions shape: {test_predictions.shape}")
print(f"Test true labels shape: {test_true_labels.shape}")

# Calculate AUC scores
print("\n" + "="*50)
print("AUC SCORES BY CLASS")
print("="*50)

# Overall AUC (macro average)
try:
    overall_auc = roc_auc_score(test_true_labels, test_predictions, average='macro')
    print(f"Overall AUC (Macro Average): {overall_auc:.4f}")
except:
    print("Could not calculate overall AUC")

# Individual class AUCs
individual_aucs = []
for i, label in enumerate(valid_labels):
    try:
        # Only calculate AUC if there are positive samples
        if test_true_labels[:, i].sum() > 0:
            auc = roc_auc_score(test_true_labels[:, i], test_predictions[:, i])
            individual_aucs.append(auc)
            print(f"{label}: {auc:.4f}")
        else:
            print(f"{label}: No positive samples in test set")
            individual_aucs.append(0.0)
    except Exception as e:
        print(f"{label}: Error calculating AUC - {str(e)}")
        individual_aucs.append(0.0)

# Weighted AUC
try:
    weighted_auc = roc_auc_score(test_true_labels, test_predictions, average='weighted')
    print(f"\nWeighted AUC: {weighted_auc:.4f}")
except:
    print("Could not calculate weighted AUC")

# Micro AUC
try:
    micro_auc = roc_auc_score(test_true_labels, test_predictions, average='micro')
    print(f"Micro AUC: {micro_auc:.4f}")
except:
    print("Could not calculate micro AUC")

print(f"\nMean Individual AUC: {np.mean([auc for auc in individual_aucs if auc > 0]):.4f}")

# Additional metrics
print("\n" + "="*50)
print("ADDITIONAL METRICS")
print("="*50)

# Convert predictions to binary (threshold = 0.5)
test_pred_binary = (test_predictions > 0.5).astype(int)

# F1 scores
try:
    f1_macro = f1_score(test_true_labels, test_pred_binary, average='macro')
    f1_micro = f1_score(test_true_labels, test_pred_binary, average='micro')
    f1_weighted = f1_score(test_true_labels, test_pred_binary, average='weighted')
    
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
except Exception as e:
    print(f"Error calculating F1 scores: {str(e)}")

# Individual F1 scores
print("\nF1 Scores by Class:")
for i, label in enumerate(valid_labels):
    try:
        if test_true_labels[:, i].sum() > 0:  # Only if there are positive samples
            f1 = f1_score(test_true_labels[:, i], test_pred_binary[:, i])
            print(f"{label}: {f1:.4f}")
    except:
        pass

print("\n" + "="*50)
print("EVALUATION COMPLETE!")
print("="*50)
print("Best model saved as 'thoracic_classifier.h5'")
print("Classification head model saved as 'classification_head.h5'")