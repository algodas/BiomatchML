import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import backend as K
import tensorflow as tf
import base64
import cv2
from skimage.morphology import skeletonize
from skimage.util import invert
from scipy.ndimage import rotate

# Fix seeds for reproducibility and force CPU
tf.random.set_seed(42)
np.random.seed(42)
tf.config.experimental.enable_op_determinism()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
MODEL_PATH = 'model/siamese_model.h5'

# Define base network for 1:1 mode (from app_1_1.py, no normalization)
def create_base_network_1_1(input_shape=(96, 96, 3)):
    base_model = MobileNetV2(include_top=False, weights=None, input_shape=input_shape)
    output = GlobalAveragePooling2D()(base_model.output)
    model = Model(base_model.input, output, name='model_1_1')
    base_model.load_weights(MODEL_PATH, by_name=True)
    for layer in model.layers:
        layer.trainable = False
    return model

# Define base network for 1:N mode (from app_1_N.py, imagenet with normalization)
def create_base_network_1_N(input_shape=(96, 96, 3)):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    output = GlobalAveragePooling2D()(base_model.output)
    output = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(output)
    model = Model(base_model.input, output, name='model_1_N')
    for layer in model.layers:
        layer.trainable = False
    return model

# Create Siamese model for 1:1 mode
def create_siamese_model_1_1(input_shape):
    input_a = Input(shape=input_shape, name="input_a")
    input_b = Input(shape=input_shape, name="input_b")
    base_network = create_base_network_1_1(input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    model = Model(inputs=[input_a, input_b], outputs=[processed_a, processed_b])
    return model

# Create Siamese model for 1:N mode
def create_siamese_model_1_N(input_shape):
    input_a = Input(shape=input_shape, name="input_a")
    input_b = Input(shape=input_shape, name="input_b")
    base_network = create_base_network_1_N(input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    model = Model(inputs=[input_a, input_b], outputs=[processed_a, processed_b])
    return model

# Initialize models
model_1_1 = create_siamese_model_1_1((96, 96, 3))
model_1_N = create_siamese_model_1_N((96, 96, 3))

# Debug: Print model summaries and sample embeddings
model_1_1.summary()
model_1_N.summary()
sample_img = tf.random.uniform((1, 96, 96, 3))
sample_emb_a_1_1, sample_emb_b_1_1 = model_1_1.predict([sample_img, sample_img])
sample_emb_a_1_N, sample_emb_b_1_N = model_1_N.predict([sample_img, sample_img])
print(f"1:1 Sample embedding a: {sample_emb_a_1_1[0][:5]}, b: {sample_emb_b_1_1[0][:5]}")
print(f"1:N Sample embedding a: {sample_emb_a_1_N[0][:5]}, b: {sample_emb_b_1_N[0][:5]}")

def preprocess(img_file, angle=0):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_bmp(img, channels=3)
    img = tf.image.resize(img, [96, 96])
    if angle != 0:
        img = tf.image.rot90(img, k=angle // 90)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def detectar_minucias(image_path, limite_maximo=80):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (96, 96))
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skeleton = skeletonize(invert(binary / 255)).astype(np.uint8)

    def distancia(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    grid_size = 4
    max_por_celula = 5
    cell_w = 96 // grid_size
    cell_h = 96 // grid_size

    keypoints_por_celula = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    keypoints_finais = []

    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] == 1:
                region = skeleton[y-1:y+2, x-1:x+2]
                total = np.sum(region) - 1
                if total == 1 or total >= 3:
                    if img[y, x] > 210 or np.std(img[y-1:y+2, x-1:x+2]) < 10:
                        continue
                    cx, cy = x // cell_w, y // cell_h
                    if cx >= grid_size or cy >= grid_size:
                        continue
                    ponto = (x, y)
                    if all(distancia(ponto, p) > 4 for p in keypoints_por_celula[cy][cx]):
                        if len(keypoints_por_celula[cy][cx]) < max_por_celula:
                            keypoints_por_celula[cy][cx].append(ponto)
                            keypoints_finais.append(ponto)

    return keypoints_finais[:limite_maximo]

def gerar_imagem_com_pontos(img_path, keypoints, color=(0, 255, 0)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (96, 96))
    for x, y in keypoints:
        cv2.circle(img, (x, y), 1, color, -1)
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode()

def generate_gradcam_heatmap(model, img_array, layer_name='block_15_add'):
    mobilenet_submodel = model.get_layer('model_1_1') if 'model_1_1' in [layer.name for layer in model.layers] else model.get_layer('model_1_N')
    grad_model = tf.keras.models.Model(
        [mobilenet_submodel.input],
        [mobilenet_submodel.get_layer(layer_name).output]
    )

    with tf.GradientTape() as tape:
        conv_outputs = grad_model(img_array)
        loss = tf.reduce_mean(conv_outputs)

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0].numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10
    return heatmap

def overlay_heatmap_on_image(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (96, 96))
    heatmap = cv2.resize(heatmap, (96, 96))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    _, buffer = cv2.imencode('.png', superimposed_img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return send_from_directory('.', 'upload.html')

@app.route('/match', methods=['POST'])
def match():
    try:
        img1 = request.files['img1']
        path1 = 'temp1.bmp'
        img1.save(path1)

        mode = request.form.get('mode', '1to1')
        show_heatmap = 'heatmap' in request.form

        input1 = preprocess(path1)
        input1 = tf.expand_dims(input1, axis=0)

        if mode == '1to1':
            # 1:1 mode from app_1_1.py
            threshold_match = 0.989
            alert_min, alert_max = (0.97, 0.989)
            threshold_no_match = 0.97

            img2 = request.files['img2']
            path2 = 'temp2.bmp'
            img2.save(path2)
            input2 = preprocess(path2)
            input2 = tf.expand_dims(input2, axis=0)
            cos_similarities = []
            for angle in [-5, 0, 5]:
                img1_rot = preprocess(path1, angle)
                img2_rot = preprocess(path2, angle)
                img1_rot = tf.expand_dims(img1_rot, axis=0)
                img2_rot = tf.expand_dims(img2_rot, axis=0)
                emb_a, emb_b = model_1_1.predict([img1_rot, img2_rot])
                cos_sim = np.dot(emb_a[0], emb_b[0]) / (np.linalg.norm(emb_a[0]) * np.linalg.norm(emb_b[0]))
                cos_similarities.append(cos_sim)
            prediction = np.mean(cos_similarities)
            print(f"Prediction (1:1) Average Cosine Similarity: {prediction}")
            min1 = len(detectar_minucias(path1))
            min2 = len(detectar_minucias(path2))
            if prediction > threshold_match and abs(min1 - min2) < 5:
                result_text = 'MATCH'
            elif alert_min <= prediction <= alert_max:
                result_text = 'INDECISO'
            else:
                result_text = 'NO MATCH'
            if alert_min <= prediction <= alert_max:
                result_text += '\n⚠️ For this comparison, a forensic expert review is recommended.'
            minucias2 = detectar_minucias(path2)
            img_minucias2_b64 = gerar_imagem_com_pontos(path2, minucias2)
            if show_heatmap:
                heatmap2 = generate_gradcam_heatmap(model_1_1, input2)
                heatmap2_b64 = overlay_heatmap_on_image(path2, heatmap2)
            else:
                heatmap2_b64 = None

        else:  # 1:N mode from app_1_N.py
            threshold_match = 0.850
            alert_min, alert_max = (0.85, 0.80)
            threshold_no_match = 0.80

            try:
                embeddings = np.load('database/templates.npy')
                with open('database/filenames.txt') as f:
                    filenames = [line.strip() for line in f]
                print(f"Database size: {len(embeddings)} embeddings, {len(filenames)} filenames")
                if not embeddings.size or not filenames:
                    result_text = 'ERROR: No embeddings or filenames available in database.'
                    top_candidate = None
                    prediction = 0.0
                    candidate_strings = []
                else:
                    embeddings_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    embeddings = embeddings / (embeddings_norms + 1e-10)
                    print(f"Database embedding norms: min={np.min(embeddings_norms):.4e}, max={np.max(embeddings_norms):.4e}")
                    cos_similarities = []
                    for angle in [-5, 0, 5]:
                        img1_rot = preprocess(path1, angle)
                        img1_rot = tf.expand_dims(img1_rot, axis=0)
                        query_emb, _ = model_1_N.predict([img1_rot, img1_rot])
                        query_emb = query_emb[0]
                        norm_a = np.linalg.norm(query_emb)
                        if norm_a == 0:
                            print(f"Warning: Query embedding has zero norm for angle {angle}")
                            sims = [0.0] * len(embeddings)
                        else:
                            query_emb = query_emb / norm_a
                            sims = []
                            for idx, emb in enumerate(embeddings):
                                sim = np.dot(query_emb, emb)
                                sims.append(sim)
                            sims = np.clip(sims, -1.0, 1.0)
                        cos_similarities.append(sims)
                    if not cos_similarities:
                        result_text = 'ERROR: No similarities computed.'
                        top_candidate = None
                        prediction = 0.0
                        candidate_strings = []
                    else:
                        avg_sims = np.mean(cos_similarities, axis=0)
                        print(f"Query embedding sample: {query_emb[:5]}")
                        print(f"Top 5 similarities: {np.sort(avg_sims)[::-1][:5]}")
                        expected_match = '2__M_Left_index_finger_CR.BMP'
                        if expected_match in filenames:
                            idx = filenames.index(expected_match)
                            print(f"Similarity for {expected_match}: {avg_sims[idx]:.4f}")
                        top_n = 5
                        sorted_indices = np.argsort(avg_sims)[::-1][:min(top_n, len(filenames))]
                        candidates = []
                        for idx in sorted_indices:
                            if idx < len(filenames):
                                candidates.append({
                                    'filename': filenames[idx],
                                    'similarity': float(avg_sims[idx]),
                                    'minutiae_count': len(detectar_minucias(os.path.join('database/imagens', filenames[idx])))
                                })
                        candidate_strings = [f"{c['filename']}: {c['similarity']:.4f}" for c in candidates]
                        print(f"Top {top_n} candidates: {candidate_strings}")
                        top_candidate = candidates[0] if candidates else None
                        prediction = avg_sims[sorted_indices[0]] if candidates else 0.0
                        if not candidates:
                            result_text = 'NO MATCH. No candidates available.'
                        elif max(avg_sims) < threshold_no_match:
                            result_text = f'NO MATCH. Closest match: {top_candidate["filename"]}'
                        else:
                            min1 = len(detectar_minucias(path1))
                            if top_candidate['similarity'] > threshold_match and abs(min1 - top_candidate['minutiae_count']) < 7:
                                result_text = f'MATCH with: {top_candidate["filename"]}'
                            elif alert_min <= top_candidate['similarity'] <= alert_max:
                                result_text = f'INDECISO with: {top_candidate["filename"]}'
                            else:
                                result_text = f'NO MATCH. Closest match: {top_candidate["filename"]}'
                            result_text += '\nTop candidates:\n' + '\n'.join(candidate_strings)
                            if alert_min <= top_candidate['similarity'] <= alert_max:
                                result_text += '\n⚠️ For this comparison, a forensic expert review is recommended.'
            except Exception as e:
                result_text = f'ERROR: Failed to process 1:N comparison: {str(e)}'
                top_candidate = None
                prediction = 0.0
                candidate_strings = []
                print(f"1:N processing error: {str(e)}")

            if top_candidate:
                minucias2 = detectar_minucias(os.path.join('database/imagens', top_candidate['filename']))
                img_minucias2_b64 = gerar_imagem_com_pontos(os.path.join('database/imagens', top_candidate['filename']), minucias2)
                if show_heatmap:
                    candidate_input = preprocess(os.path.join('database/imagens', top_candidate['filename']))
                    candidate_input = tf.expand_dims(candidate_input, axis=0)
                    heatmap2 = generate_gradcam_heatmap(model_1_N, candidate_input)
                    heatmap2_b64 = overlay_heatmap_on_image(os.path.join('database/imagens', top_candidate['filename']), heatmap2)
                else:
                    heatmap2_b64 = None
            else:
                minucias2 = []
                img_minucias2_b64 = None
                heatmap2_b64 = None

        minucias1 = detectar_minucias(path1)
        img_minucias1_b64 = gerar_imagem_com_pontos(path1, minucias1)
        heatmap1_b64 = None
        if show_heatmap:
            heatmap1 = generate_gradcam_heatmap(model_1_1 if mode == '1to1' else model_1_N, input1)
            heatmap1_b64 = overlay_heatmap_on_image(path1, heatmap1)

        return render_template_string("""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <title>Comparison Result</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; background: #f5f6fa; margin: 0; padding: 0; }
    .result-container { background: white; max-width: 640px; margin: 60px auto; padding: 30px; border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    h1 { color: #2f3640; margin-bottom: 20px; }
    p { font-size: 16px; margin: 10px 0; }
    .match { color: green; font-weight: bold; font-size: 18px; }
    .nomatch { color: red; font-weight: bold; font-size: 18px; }
    img { width: 300px; margin: 15px; border: 1px solid #ccc; border-radius: 6px; }
    .btn { display: inline-block; margin-top: 20px; background: #0984e3; color: white; padding: 10px 18px; border-radius: 6px; text-decoration: none; font-weight: bold; }
    .btn:hover { background: #74b9ff; }
  </style>
</head>
<body>
  <div class='result-container'>
    <h1>Comparison Result</h1>
    <p><strong>Score:</strong> {{ score }}</p>
    <p class='{{ 'match' if resultado.startswith('MATCH') else 'nomatch' if resultado.startswith('NO MATCH') else '' }}'>Result: {{ resultado }}</p>
    {% if heatmap1 %}<h3>Grad-CAM Heatmap - Query</h3><img src='data:image/png;base64,{{ heatmap1 }}'>{% endif %}
    {% if heatmap2 %}<h3>Grad-CAM Heatmap - Candidate</h3><img src='data:image/png;base64,{{ heatmap2 }}'>{% endif %}
    <h3>🧬 Minutiae - Query</h3><img src='data:image/png;base64,{{ img_minucias1 }}'>
    {% if img_minucias2 %}<h3>🧬 Minutiae - Candidate</h3><img src='data:image/png;base64,{{ img_minucias2 }}'>{% endif %}
    <a class='btn' href='/'>↩ Back</a>
  </div>
</body>
</html>""",
        score=round(float(prediction), 4),
        resultado=result_text,
        heatmap1=heatmap1_b64,
        heatmap2=heatmap2_b64,
        img_minucias1=img_minucias1_b64,
        img_minucias2=img_minucias2_b64)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
