import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Lambda
from tensorflow.keras.applications import MobileNetV2
from scipy.ndimage import rotate

# Fix seeds for reproducibility and force CPU
tf.random.set_seed(42)
np.random.seed(42)
tf.config.experimental.enable_op_determinism()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Preprocess function (match app.py)
def preprocess(img_file, angle=0):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_bmp(img, channels=3)
    img = tf.image.resize(img, [96, 96])
    if angle != 0:
        img = tf.image.rot90(img, k=angle // 90)
    img = tf.cast(img, tf.float32) / 255.0
    return img

# Create base network (use imagenet weights)
def create_base_network(input_shape=(96, 96, 3)):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    output = GlobalAveragePooling2D()(base_model.output)
    output = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(output)  # Normalize embeddings
    model = Model(base_model.input, output, name='model_1')
    for layer in model.layers:
        layer.trainable = False
    return model

# Create Siamese model
def create_siamese_model(input_shape):
    input_a = Input(shape=input_shape, name="input_a")
    input_b = Input(shape=input_shape, name="input_b")
    base_network = create_base_network(input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    model = Model(inputs=[input_a, input_b], outputs=[processed_a, processed_b])
    return model

# Initialize model
model = create_siamese_model((96, 96, 3))

# Paths
image_dir = 'database/imagens'
output_embeddings = 'database/templates.npy'
output_filenames = 'database/filenames.txt'

# Generate embeddings
embeddings = []
filenames = []

for fname in os.listdir(image_dir):
    if fname.lower().endswith('.bmp'):
        path = os.path.join(image_dir, fname)
        input_img = preprocess(path)
        input_img = tf.expand_dims(input_img, axis=0)
        emb_a, _ = model.predict([input_img, input_img])
        norm_a = np.linalg.norm(emb_a[0])
        if norm_a < 1e-10:
            print(f"Warning: Embedding for {fname} has near-zero norm ({norm_a:.4e})")
            continue  # Skip invalid embeddings
        embeddings.append(emb_a[0])
        filenames.append(fname)
        print(f"Processed: {fname}, Embedding sample: {emb_a[0][:5]}, Norm: {norm_a:.6f}")

# Save embeddings and filenames
np.save(output_embeddings, np.array(embeddings))
with open(output_filenames, 'w') as f:
    f.write('\n'.join(filenames))

print(f"Saved {len(embeddings)} embeddings to {output_embeddings}")
print(f"Saved filenames to {output_filenames}")
