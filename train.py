import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
from PIL import Image
import random

def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

def preprocess_image(path, target_shape=(96, 96)):
    img = Image.open(path).convert('RGB')
    img = img.resize(target_shape)
    img = np.array(img).astype("float32") / 255.0
    return img

def load_pairs(data_dir, input_shape=(96, 96), max_pairs=2000):
    files = [f for f in os.listdir(data_dir) if f.endswith('.BMP')]
    identities = {}

    for f in files:
        identity = f.split('__')[0] if '__' in f else f.split('_')[0]
        identities.setdefault(identity, []).append(os.path.join(data_dir, f))

    pairs = []
    labels = []

    keys = list(identities.keys())

    for _ in tqdm(range(max_pairs)):
        # Positive pair
        c = random.choice(keys)
        if len(identities[c]) < 2:
            continue
        img1, img2 = random.sample(identities[c], 2)
        pairs.append([preprocess_image(img1), preprocess_image(img2)])
        labels.append(1)

        # Negative pair
        c1, c2 = random.sample(keys, 2)
        if len(identities[c1]) < 1 or len(identities[c2]) < 1:
            continue
        img1 = random.choice(identities[c1])
        img2 = random.choice(identities[c2])
        pairs.append([preprocess_image(img1), preprocess_image(img2)])
        labels.append(0)

    pairs = np.array(pairs)
    return [pairs[:, 0], pairs[:, 1]], np.array(labels)

def build_siamese_model(input_shape):
    def build_base_network(input_shape):
        inp = Input(shape=input_shape)
        x = Conv2D(64, (3, 3), activation='relu')(inp)
        x = MaxPooling2D()(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        return Model(inp, x)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    base_network = build_base_network(input_shape)

    encoded_a = base_network(input_a)
    encoded_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=(1,))([encoded_a, encoded_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model

if __name__ == "__main__":
    data_path = '/home/ubuntu/SOCOFing/Real'
    input_shape = (96, 96, 3)
    epochs = 5

    print("🔍 Carregando pares de imagens...")
    X, y = load_pairs(data_path, input_shape=input_shape[:2], max_pairs=2000)

    print("🚀 Iniciando treinamento...")
    model = build_siamese_model(input_shape)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001))
    model.fit([X[0], X[1]], y, batch_size=32, epochs=epochs)

    os.makedirs('model', exist_ok=True)
    model.save('model/siamese_model.h5')
    print("✅ Modelo salvo em model/siamese_model.h5")
 
