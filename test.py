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
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import glob


# Dataset paths
BASE_DIR = "./dataset_balanced/"
CSV_PATH = os.path.join(BASE_DIR, "new_labels.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "new_images", "new_images")

# Verify paths exist
if not os.path.exists(CSV_PATH):
    print(f"CSV file not found at: {CSV_PATH}")
if not os.path.exists(IMAGE_DIR):
    print(f"Image directory not found at: {IMAGE_DIR}")

# Get image files
IMAGE_FILES = {os.path.basename(f): f for f in glob.glob(os.path.join(IMAGE_DIR, "*.png"))}
print(f"Total images found: {len(IMAGE_FILES)}")

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"CSV shape: {df.shape}")
print(f"CSV columns: {df.columns.tolist()}")

# Check if the correct columns exist
if 'Image Index' not in df.columns:
    print("Warning: 'Image Index' column not found. Available columns:", df.columns.tolist())
if 'Finding Labels' not in df.columns:
    print("Warning: 'Finding Labels' column not found. Available columns:", df.columns.tolist())

# Keep only relevant columns
df = df[['Image Index', 'Finding Labels']]

# Convert multi-labels into lists
df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|') if isinstance(x, str) else [])

# Filter only images that exist in the image directory
df_filtered = df[df['Image Index'].isin(IMAGE_FILES.keys())].copy()
print(f"Images after filtering: {len(df_filtered)}")

if len(df_filtered) == 0:
    print("No matching images found! Check if image names in CSV match actual image files.")
    print("Sample CSV image names:", df['Image Index'].head().tolist())
    print("Sample actual image files:", list(IMAGE_FILES.keys())[:5])

# Binarize the labels
mlb = MultiLabelBinarizer()
binarized_labels = mlb.fit_transform(df_filtered['Finding Labels'])

# Convert binarized labels into a DataFrame with column names
valid_labels = mlb.classes_
print(f"Valid labels: {valid_labels}")

label_columns = valid_labels
df_labels = pd.DataFrame(binarized_labels, columns=label_columns)

# Concatenate with the original DataFrame
df_filtered = pd.concat([df_filtered[['Image Index']], df_labels], axis=1)

# Check label distribution
print("Label distribution:")
print(df_labels.sum(axis=0))

# Display some examples
print("\nSample data:")
print(df_filtered.head())

# Check for Pneumonia examples
pneumonia_examples = df[df['Finding Labels'].apply(lambda x: 'Pneumonia' in x)]
print(f"\nPneumonia examples found: {len(pneumonia_examples)}")
print(pneumonia_examples.head())

# Show label combinations
print("\nLabel combinations:")
label_combinations = df_filtered['Finding Labels'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
print(label_combinations.value_counts().head(10))

# Prepare input (X) and labels (y)
X = df_filtered['Image Index'].values  # Image filenames
y = df_filtered[valid_labels].values   # Multi-label one-hot encoding

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None  # Remove stratify for multilabel
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=None
)

print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Create DataFrames for generators
train_df = pd.DataFrame({'Image Index': X_train})
train_df = pd.concat([train_df, pd.DataFrame(y_train, columns=valid_labels)], axis=1)

val_df = pd.DataFrame({'Image Index': X_val})
val_df = pd.concat([val_df, pd.DataFrame(y_val, columns=valid_labels)], axis=1)

test_df = pd.DataFrame({'Image Index': X_test})
test_df = pd.concat([test_df, pd.DataFrame(y_test, columns=valid_labels)], axis=1)

# Augmentation settings for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
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

# Train generator with augmentation
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=IMAGE_DIR,  # Correct path to images
    x_col="Image Index",
    y_col=list(valid_labels),  # Convert to list
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True
)

# Validation generator (no augmentation)
val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=IMAGE_DIR,  # Correct path to images
    x_col="Image Index",
    y_col=list(valid_labels),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False
)

# Test generator (for final evaluation)
test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=IMAGE_DIR,  # Correct path to images
    x_col="Image Index",
    y_col=list(valid_labels),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False
)

print("Data generators created successfully!")

# Load ResNet-50 with pretrained weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Add new classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x_class = Dense(128, activation='relu')(x)
x_class = Dropout(0.3)(x_class)
output_class = Dense(len(valid_labels), activation='sigmoid', name='classification_output')(x_class)

# Define and compile model
model = Model(inputs=base_model.input, outputs=output_class)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy', 'AUC']
)

print("Model created and compiled!")
print(f"Model will predict {len(valid_labels)} classes: {valid_labels}")

# Callbacks
checkpoint = ModelCheckpoint(
    'classification_head.h5', 
    monitor='val_auc', 
    mode='max', 
    save_best_only=True, 
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=2, 
    verbose=1, 
    min_lr=1e-6
)

callbacks = [checkpoint, reduce_lr]

# Train the classification head only
print("Starting initial training (frozen base model)...")
history = model.fit(
    train_generator,
    epochs=5,  
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks,
    verbose=1
)

# Unfreeze last 10 layers of ResNet-50 for fine-tuning
print("Unfreezing last 10 layers for fine-tuning...")
for layer in base_model.layers[-10:]:
    layer.trainable = True

finetune_checkpoint = ModelCheckpoint(
    'baseline_resnet50.h5', 
    monitor='val_auc', 
    mode='max', 
    save_best_only=True, 
    verbose=1
)
finetune_callbacks = [finetune_checkpoint, reduce_lr]

# Recompile with a smaller learning rate for fine-tuning
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    metrics=['accuracy', 'AUC']
)

# Fine-tuning
print("Starting fine-tuning...")
history_finetune = model.fit(
    train_generator,
    epochs=5, 
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=finetune_callbacks,
    verbose=1
)

# Load the best model
print("Loading best model...")
model = load_model('baseline_resnet50.h5')

# Test prediction on a batch
print("Testing prediction...")
test_generator.reset()  # Reset generator to start from beginning
batch_images, batch_labels = next(iter(test_generator))

# Predict on the batch
batch_preds = model.predict(batch_images)

# Choose a random image from the batch
idx = np.random.randint(len(batch_images))
img = batch_images[idx]
true_label = batch_labels[idx]
pred_label = (batch_preds[idx] > 0.19).astype(int)  # Threshold for prediction

# Convert one-hot encoded labels back to class names
true_label_names = [valid_labels[i] for i in range(len(valid_labels)) if true_label[i] == 1]
pred_label_names = [valid_labels[i] for i in range(len(valid_labels)) if pred_label[i] == 1]

# Remove "No Finding" if there are other predicted labels
if "No Finding" in pred_label_names and len(pred_label_names) > 1:
    pred_label_names.remove("No Finding")

# Show the image
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.title(f"Predicted: {', '.join(pred_label_names) if pred_label_names else 'None'}\nActual: {', '.join(true_label_names) if true_label_names else 'None'}")
plt.show()

# Save models
print("Saving models...")
model.save('baseline_model_resnet50.h5')  # Fixed typo in filename
model.save('baseline_model_resnet50.keras')

print("Evaluating on test set...")
# Reset test generator for full evaluation
test_generator.reset()

# Predict probabilities for test set
y_pred = model.predict(test_generator, verbose=1)

# Ensure we have the correct number of predictions
if len(y_pred) != len(y_test):
    print(f"Warning: Prediction length ({len(y_pred)}) doesn't match test length ({len(y_test)})")
    # Truncate to minimum length
    min_len = min(len(y_pred), len(y_test))
    y_pred = y_pred[:min_len]
    y_test_eval = y_test[:min_len]
else:
    y_test_eval = y_test

# Compute Precision-Recall and AUC-PR for each class
precision = dict()
recall = dict()
auc_pr = dict()

for i in range(len(valid_labels)):
    precision[i], recall[i], _ = precision_recall_curve(y_test_eval[:, i], y_pred[:, i])
    auc_pr[i] = auc(recall[i], precision[i])

# Plot Precision-Recall curve for each class
plt.figure(figsize=(15, 10))
for i in range(len(valid_labels)):
    plt.plot(recall[i], precision[i], lw=2, label=f'{valid_labels[i]} (AUC-PR = {auc_pr[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for All Classes')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nModel Training Complete!")
print(f"Classes: {len(valid_labels)}")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")
print("\nAUC-PR scores by class:")
for i, label in enumerate(valid_labels):
    print(f"{label}: {auc_pr[i]:.3f}")

print(f"\nMean AUC-PR: {np.mean(list(auc_pr.values())):.3f}")