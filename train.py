import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Generator for Batch Processing
class SiameseDataGenerator(Sequence):
    def __init__(self, pair_paths, labels, batch_size=16, input_shape=(96, 96, 3), shuffle=True):
        self.pair_paths = pair_paths  # List of (path1, path2) tuples
        self.labels = labels
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.pair_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        X1 = []
        X2 = []
        y = []

        for idx in indexes:
            img1_path, img2_path = self.pair_paths[idx]
            img1 = self._preprocess_image(img1_path)
            img2 = self._preprocess_image(img2_path)
            X1.append(img1)
            X2.append(img2)
            y.append(self.labels[idx])

        return (np.array(X1), np.array(X2)), np.array(y)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pair_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.input_shape[:2])
        img_tensor = img_resized / 255.0
        return img_tensor

# Define the base network
def create_base_network(input_shape=(96, 96, 3)):
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    output = GlobalAveragePooling2D()(base_model.output)
    return Model(base_model.input, output)

# Create Siamese model
def create_siamese_model(input_shape):
    input_a = Input(shape=input_shape, name="input_a")
    input_b = Input(shape=input_shape, name="input_b")
    
    base_network = create_base_network(input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True)))([processed_a, processed_b])
    output = Dense(1, activation="sigmoid")(distance)
    
    model = Model(inputs=[input_a, input_b], outputs=output)
    return model

# Load and prepare dataset
def load_dataset(dataset_path):
    pair_paths = []  # List of (path1, path2) tuples
    labels = []  
    
    # Debug: List available directories
    logger.info(f"Contents of {dataset_path}: {os.listdir(dataset_path)}")
    
    # Collect all image paths from NOVO and its subfolders
    novo_dir = os.path.join(dataset_path, "NOVO")
    if not os.path.exists(novo_dir):
        logger.error(f"Directory 'NOVO' not found in {dataset_path}")
        available_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        logger.error(f"Available directories: {available_dirs}")
        raise FileNotFoundError(f"Directory 'NOVO' not found in {dataset_path}. Available: {available_dirs}")
    
    # Recursively collect all .bmp files
    all_images = []
    for root, _, files in os.walk(novo_dir):
        for file in files:
            if file.lower().endswith('.bmp'):  # Specifically target .bmp files
                all_images.append(os.path.join(root, file))
    
    if not all_images:
        logger.error("No .bmp images found in 'NOVO' or its subfolders.")
        raise ValueError("No .bmp images found in 'NOVO' or its subfolders.")
    
    logger.info(f"Found {len(all_images)} .bmp images in 'NOVO' and its subfolders.")
    
    # Generate match pairs (assuming subfolders or similar filenames indicate matches)
    image_dict = {}  # Group by directory or base name
    for img_path in all_images:
        dir_name = os.path.basename(os.path.dirname(img_path)) or "root"
        base_name = os.path.splitext(os.path.basename(img_path))[0].split('_')[0]  # Extract base name
        key = f"{dir_name}_{base_name}"
        if key not in image_dict:
            image_dict[key] = []
        image_dict[key].append(img_path)
    
    for key, img_list in image_dict.items():
        if len(img_list) > 1:  # Ensure at least two images for a match
            for i in range(len(img_list)):
                for j in range(i + 1, min(i + 5, len(img_list))):  # Limit to 5 pairs per group
                    pair_paths.append((img_list[i], img_list[j]))
                    labels.append(1)  # Match
    
    # Generate non-match pairs (random pairs from different base names)
    base_names = list(image_dict.keys())
    all_images_list = all_images  # For indexing
    for i in range(len(all_images_list)):
        for j in range(i + 1, min(i + 5, len(all_images_list))):
            img1_base = os.path.splitext(os.path.basename(all_images_list[i]))[0].split('_')[0]
            img2_base = os.path.splitext(os.path.basename(all_images_list[j]))[0].split('_')[0]
            if img1_base != img2_base:  # Ensure different base names for non-match
                pair_paths.append((all_images_list[i], all_images_list[j]))
                labels.append(0)  # Non-match
    
    # Limit pairs to avoid memory issues
    max_pairs = 20000  # Adjust based on dataset size and memory; increase if needed for better training
    if len(pair_paths) > max_pairs:
        indices = np.random.choice(len(pair_paths), max_pairs, replace=False)
        pair_paths = [pair_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    if not pair_paths:
        logger.error("No valid image pairs generated.")
        raise ValueError("No valid image pairs generated.")
    
    logger.info(f"Generated {len(labels)} pairs: {np.bincount(labels)} (match, non-match).")
    return pair_paths, labels

# Main training function
def train_model():
    # Load dataset
    dataset_path = "/kaggle/input/dataset"
    pair_paths, labels = load_dataset(dataset_path)
    
    # Split dataset
    train_pairs, val_pairs, train_labels, val_labels = train_test_split(
        pair_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create data generators
    train_generator = SiameseDataGenerator(train_pairs, train_labels, batch_size=16)
    val_generator = SiameseDataGenerator(val_pairs, val_labels, batch_size=16, shuffle=False)
    
    # Create and compile model
    model = create_siamese_model(input_shape=(96, 96, 3))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    logger.info("Model compiled successfully.")
    model.summary()
    
    # Add early stopping to prevent overfitting and save resources
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    # Save model
    os.makedirs("model", exist_ok=True)
    model.save("siamese_model.h5")
    logger.info("Model saved to model/siamese_model.h5")

if __name__ == "__main__":
    train_model()
