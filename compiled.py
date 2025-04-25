import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_recall_curve, auc

# ======================================================================

BASE_DIR = "D:/dataset"
CSV_PATH = os.path.join(BASE_DIR, "Data_Entry_2017.csv")

print(CSV_PATH)

# Gather all image files across the 12 folders
IMAGE_DIR = os.path.join(BASE_DIR, "images")
IMAGE_FILES = {os.path.basename(f): f for f in glob.glob(os.path.join(IMAGE_DIR, "*.png"))}

print(IMAGE_FILES)

print(f"Total images found: {len(IMAGE_FILES)}")


# ======================================================================


# Load CSV and extract relevant columns
df = pd.read_csv(CSV_PATH)
df = df[['Image Index', 'Finding Labels']]
df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))

# Define the disease labels you're interested in
labels = ['Pneumonia', 'Cardiomegaly', 'Edema', 'Emphysema', 'Effusion', 'Infiltration', 'Atelectasis']

# One-hot encode the disease labels
for label in labels:
    df[label] = df['Finding Labels'].apply(lambda x: 1 if label in x else 0)

# Compute the total number of labels per row (helps identify "No Finding")
df['has_label'] = df[labels].sum(axis=1)

# Split into disease and "No Finding" groups
disease_df = df[df['has_label'] > 0]
no_finding_df = df[df['has_label'] == 0]

# Downsample "No Finding" rows to balance the dataset
downsample_ratio = 0.3  # You can tune this ratio
desired_no_finding = int(len(disease_df) * downsample_ratio)
downsampled_no_finding_df = no_finding_df.sample(n=desired_no_finding, random_state=42)

# Combine and shuffle the final dataset
balanced_df = pd.concat([disease_df, downsampled_no_finding_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Train-validation split
train_df, val_df = train_test_split(balanced_df, test_size=0.2, random_state=42)


# =======================================================================


datagen = ImageDataGenerator(rescale=1./255)

# With augmentation
train_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True
)

# Add the 'path' column to train_df and val_df
train_df['path'] = train_df['Image Index'].map(IMAGE_FILES)
val_df['path'] = val_df['Image Index'].map(IMAGE_FILES)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,  # Set to None since paths are absolute
    x_col='path',
    y_col=labels,
    target_size=(224, 224),
    batch_size=16,
    class_mode='raw',
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,  # Set to None since paths are absolute
    x_col='path',
    y_col=labels,
    target_size=(224, 224),
    batch_size=16,
    class_mode='raw',
    shuffle=False
)


# =======================================================================


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze base initially
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(len(labels), activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# =======================================================================


# Compute per-class weights (inverse frequency)
class_freq = train_df[labels].sum()
class_weights = 1. / (class_freq + 1e-6)
class_weights = class_weights / class_weights.sum()  # normalize to sum to 1
class_weights_tensor = K.constant(class_weights.values)  # convert to tensor

# Updated focal loss with per-class alpha
def focal_loss(gamma=2., alpha=class_weights_tensor):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')  # Cast y_true to float32
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # Clip predictions to avoid log(0)
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return K.mean(alpha_t * K.pow(1 - p_t, gamma) * bce, axis=-1)  # Mean over batch
    return loss

# Unfreeze top layers of base model
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Compile model
model.compile(
    optimizer=Adam(1e-4),
    loss=focal_loss(),  # custom loss with per-class balancing
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("thoracic_classifierV5.h5", save_best_only=True)
]

# Train
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    callbacks=callbacks
)


# =======================================================================


# True and predicted labels
y_true = val_gen.labels
y_pred = model.predict(val_gen)

# Compute Precision-Recall and AUC-PR for each class
precision = dict()
recall = dict()
auc_pr = dict()

for i in range(len(labels)):
    precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
    auc_pr[i] = auc(recall[i], precision[i])

# Plot
plt.figure(figsize=(12, 8))
for i in range(len(labels)):
    plt.plot(recall[i], precision[i], lw=2, label=f'{labels[i]} (AUC-PR = {auc_pr[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (One-vs-All)')
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.show()


# =======================================================================


model.save('thoracic_classifierV5.h5')  # Saves as HDF5 format
model.save('thoracic_classifierV5.keras')  # Saves as Keras format
