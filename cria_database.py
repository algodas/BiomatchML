# create_database.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import backend as K

# Define the base network (match original structure)
def create_base_network(input_shape=(96, 96, 3)):
    base_model = MobileNetV2(include_top=False, weights=None, input_shape=input_shape)
    output = GlobalAveragePooling2D()(base_model.output)
    return Model(base_model.input, output)

# Load base model with original weights
base_model = create_base_network((96, 96, 3))
base_model.load_weights('model/siamese_model.h5', by_name=True)  # Load weights up to base_network

image_dir = 'database/imagens'
filenames = []
embeddings = []

for fname in os.listdir(image_dir):
    if fname.lower().endswith('.bmp'):
        path = os.path.join(image_dir, fname)
        img = tf.io.read_file(path)
        img = tf.image.decode_bmp(img, channels=3)
        img = tf.image.resize(img, [96, 96])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)
        emb = base_model.predict(img)[0]
        embeddings.append(emb)
        filenames.append(fname)

np.save('database/templates.npy', np.array(embeddings))
with open('database/filenames.txt', 'w') as f:
    for name in filenames:
        f.write(name + '\n')
