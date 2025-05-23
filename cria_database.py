# create_database.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from app import preprocess, euclidean_distance

model = load_model('model/siamese_model.h5', compile=False, custom_objects={'euclidean_distance': euclidean_distance})
base_model = model.get_layer('functional')

image_dir = 'database/imagens'
filenames = []
embeddings = []

for fname in os.listdir(image_dir):
    if fname.lower().endswith('.bmp'):
        path = os.path.join(image_dir, fname)
        img = preprocess(path)
        img = tf.expand_dims(img, axis=0)
        emb = base_model.predict(img)[0]
        embeddings.append(emb)
        filenames.append(fname)

np.save('database/templates.npy', np.array(embeddings))
with open('database/filenames.txt', 'w') as f:
    for name in filenames:
        f.write(name + '\n')
