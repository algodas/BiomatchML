import os
import uuid
import logging
import tempfile
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template_string, render_template
from flask_wtf import CSRFProtect
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

logging.basicConfig(level=logging.INFO)

# Fix seeds for reproducibility and force CPU
tf.random.set_seed(42)
np.random.seed(42)
tf.config.experimental.enable_op_determinism()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY') or os.urandom(32)
app.config['WTF_CSRF_TIME_LIMIT'] = 3600
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
csrf = CSRFProtect(app)
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

def _is_valid_bmp(file_obj) -> bool:
    """Validate BMP magic bytes (must start with 'BM')."""
    header = file_obj.read(2)
    file_obj.seek(0)
    return header == b'BM'

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/match', methods=['POST'])
def match():
    req_id = uuid.uuid4().hex
    path1 = os.path.join(tempfile.gettempdir(), f'bio_{req_id}_1.bmp')
    path2 = os.path.join(tempfile.gettempdir(), f'bio_{req_id}_2.bmp')
    try:
        img1 = request.files['img1']
        if not img1.filename.lower().endswith('.bmp'):
            return jsonify({'error': 'Formato inválido. Apenas arquivos BMP são permitidos.'}), 400
        if not _is_valid_bmp(img1):
            return jsonify({'error': 'Arquivo não é um BMP válido.'}), 400
        if len(img1.read()) > 3 * 1024 * 1024:
            return jsonify({'error': 'Arquivo muito grande. Máximo permitido: 3MB.'}), 400
        img1.seek(0)


        img1.save(path1)

        mode = request.form.get('mode', '1to1')
        show_heatmap = 'heatmap' in request.form

        input1 = preprocess(path1)
        input1 = tf.expand_dims(input1, axis=0)

        if mode == '1to1':
            # 1:1 mode from app_1_1.py
            threshold_match = 0.989
            alert_min, alert_max = (0.985, 0.989)
            threshold_no_match = 0.984

            img2 = request.files['img2']
            if not img2.filename.lower().endswith('.bmp'):
                return jsonify({'error': 'Formato inválido. Apenas arquivos BMP são permitidos.'}), 400
            if not _is_valid_bmp(img2):
                return jsonify({'error': 'Arquivo não é um BMP válido.'}), 400
            if len(img2.read()) > 3 * 1024 * 1024:
                return jsonify({'error': 'Arquivo muito grande. Máximo permitido: 3MB.'}), 400
            img2.seek(0)



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
  <meta name='viewport' content='width=device-width, initial-scale=1.0'>
  <title>Comparison Result – BiomatchML</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #f0f2f5;
      --surface: #ffffff;
      --border: #e2e5ea;
      --primary: #1a56db;
      --text: #1a1d23;
      --muted: #6b7280;
      --header-bg: #0f172a;
      --match-bg: #f0fdf4;
      --match-border: #bbf7d0;
      --match-text: #15803d;
      --nomatch-bg: #fef2f2;
      --nomatch-border: #fecaca;
      --nomatch-text: #b91c1c;
      --undecided-bg: #fffbeb;
      --undecided-border: #fde68a;
      --undecided-text: #92400e;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      background: var(--header-bg);
      color: white;
      padding: 0 32px;
      height: 54px;
      display: flex;
      align-items: center;
      gap: 12px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    }

    .logo { display: flex; align-items: center; gap: 10px; }

    .logo-icon {
      width: 30px; height: 30px;
      background: var(--primary);
      border-radius: 6px;
      display: flex; align-items: center; justify-content: center;
    }

    header h1 { font-size: 15px; font-weight: 600; }

    .header-badge {
      font-size: 11px; color: #94a3b8;
      padding: 2px 8px;
      border: 1px solid #334155;
      border-radius: 4px;
    }

    main {
      flex: 1;
      display: flex;
      justify-content: center;
      padding: 36px 16px 48px;
    }

    .page {
      width: 100%;
      max-width: 680px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .card {
      background: var(--surface);
      border-radius: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 14px rgba(0,0,0,0.05);
      overflow: hidden;
    }

    .card-header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .card-header svg { color: var(--muted); flex-shrink: 0; }

    .card-header h2 {
      font-size: 14px;
      font-weight: 600;
      color: var(--text);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .card-body { padding: 20px 24px; }

    /* Result verdict */
    .verdict {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 16px 20px;
      border-radius: 10px;
      border: 1px solid;
    }

    .verdict.match     { background: var(--match-bg);     border-color: var(--match-border);     color: var(--match-text); }
    .verdict.nomatch   { background: var(--nomatch-bg);   border-color: var(--nomatch-border);   color: var(--nomatch-text); }
    .verdict.undecided { background: var(--undecided-bg); border-color: var(--undecided-border); color: var(--undecided-text); }

    .verdict-icon {
      width: 36px; height: 36px;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0;
    }

    .verdict.match     .verdict-icon { background: #dcfce7; }
    .verdict.nomatch   .verdict-icon { background: #fee2e2; }
    .verdict.undecided .verdict-icon { background: #fef3c7; }

    .verdict-label {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }

    .verdict-detail {
      font-size: 12px;
      margin-top: 2px;
      opacity: 0.8;
      white-space: pre-wrap;
    }

    /* Score row */
    .score-row {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-top: 16px;
    }

    .score-label {
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      white-space: nowrap;
    }

    .score-bar-wrap {
      flex: 1;
      height: 6px;
      background: #e5e7eb;
      border-radius: 3px;
      overflow: hidden;
    }

    .score-bar {
      height: 100%;
      border-radius: 3px;
      background: var(--primary);
      transition: width 0.6s ease;
    }

    .score-value {
      font-size: 14px;
      font-weight: 700;
      color: var(--text);
      white-space: nowrap;
    }

    /* Image grid */
    .img-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }

    @media (max-width: 480px) { .img-grid { grid-template-columns: 1fr; } }

    .img-block { display: flex; flex-direction: column; gap: 8px; }

    .img-block-label {
      font-size: 11px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.07em;
    }

    .img-block img {
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--border);
      display: block;
      image-rendering: pixelated;
    }

    /* Back button */
    .btn-back {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 9px 18px;
      background: var(--primary);
      color: white;
      text-decoration: none;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      transition: background 0.2s;
    }

    .btn-back:hover { background: #1342b8; }
  </style>
</head>
<body>

  <header>
    <div class='logo'>
      <div class='logo-icon'>
        <svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
          <path d='M12 2a9.96 9.96 0 0 0-6.5 2.4'/><path d='M18.5 4.4A9.97 9.97 0 0 1 22 12c0 1.9-.5 3.7-1.4 5.2'/><path d='M4 7.2A9.94 9.94 0 0 0 2 12a9.9 9.9 0 0 0 2.2 6.2'/><path d='M12 8a4 4 0 0 0-4 4 4 4 0 0 0 1.1 2.7'/><path d='M16 12a4 4 0 0 0-1-2.7'/><path d='M8 17.5A7 7 0 0 0 12 19a7 7 0 0 0 5.4-2.5'/><path d='M12 12v4'/>
        </svg>
      </div>
      <h1>BiomatchML</h1>
    </div>
    <span class='header-badge'>Comparison Result</span>
  </header>

  <main>
    <div class='page'>

      <!-- Verdict card -->
      <div class='card'>
        <div class='card-header'>
          <svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M22 11.08V12a10 10 0 1 1-5.93-9.14'/><polyline points='22 4 12 14.01 9 11.01'/></svg>
          <h2>Verdict</h2>
        </div>
        <div class='card-body'>
          {% if resultado.startswith('MATCH') %}
          <div class='verdict match'>
            <div class='verdict-icon'>
              <svg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='#15803d' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><polyline points='20 6 9 17 4 12'/></svg>
            </div>
            <div>
              <div class='verdict-label'>MATCH</div>
              <div class='verdict-detail'>{{ resultado[5:] if resultado|length > 5 else '' }}</div>
            </div>
          </div>
          {% elif resultado.startswith('NO MATCH') %}
          <div class='verdict nomatch'>
            <div class='verdict-icon'>
              <svg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='#b91c1c' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><line x1='18' y1='6' x2='6' y2='18'/><line x1='6' y1='6' x2='18' y2='18'/></svg>
            </div>
            <div>
              <div class='verdict-label'>NO MATCH</div>
              <div class='verdict-detail'>{{ resultado[8:] if resultado|length > 8 else '' }}</div>
            </div>
          </div>
          {% else %}
          <div class='verdict undecided'>
            <div class='verdict-icon'>
              <svg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='#92400e' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><line x1='12' y1='8' x2='12' y2='12'/><line x1='12' y1='16' x2='12.01' y2='16'/></svg>
            </div>
            <div>
              <div class='verdict-label'>INCONCLUSIVE</div>
              <div class='verdict-detail'>{{ resultado }}</div>
            </div>
          </div>
          {% endif %}

          <div class='score-row'>
            <span class='score-label'>Score</span>
            <div class='score-bar-wrap'>
              <div class='score-bar' style='width: {{ [score * 100, 100]|min }}%'></div>
            </div>
            <span class='score-value'>{{ score }}</span>
          </div>
        </div>
      </div>

      <!-- Minutiae card -->
      <div class='card'>
        <div class='card-header'>
          <svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='8'/><path d='m21 21-4.35-4.35'/></svg>
          <h2>Minutiae Analysis</h2>
        </div>
        <div class='card-body'>
          <div class='img-grid'>
            <div class='img-block'>
              <span class='img-block-label'>Query</span>
              <img src='data:image/png;base64,{{ img_minucias1 }}' alt='Minutiae – Query'>
            </div>
            {% if img_minucias2 %}
            <div class='img-block'>
              <span class='img-block-label'>Candidate</span>
              <img src='data:image/png;base64,{{ img_minucias2 }}' alt='Minutiae – Candidate'>
            </div>
            {% endif %}
          </div>
        </div>
      </div>

      <!-- Grad-CAM card -->
      {% if heatmap1 or heatmap2 %}
      <div class='card'>
        <div class='card-header'>
          <svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z'/><circle cx='12' cy='12' r='3'/></svg>
          <h2>Grad-CAM Heatmap</h2>
        </div>
        <div class='card-body'>
          <div class='img-grid'>
            {% if heatmap1 %}
            <div class='img-block'>
              <span class='img-block-label'>Query</span>
              <img src='data:image/png;base64,{{ heatmap1 }}' alt='Grad-CAM – Query'>
            </div>
            {% endif %}
            {% if heatmap2 %}
            <div class='img-block'>
              <span class='img-block-label'>Candidate</span>
              <img src='data:image/png;base64,{{ heatmap2 }}' alt='Grad-CAM – Candidate'>
            </div>
            {% endif %}
          </div>
        </div>
      </div>
      {% endif %}

      <div>
        <a class='btn-back' href='/'>
          <svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><polyline points='15 18 9 12 15 6'/></svg>
          New Comparison
        </a>
      </div>

    </div>
  </main>

</body>
</html>""",
        score=round(float(prediction), 4),
        resultado=result_text,
        heatmap1=heatmap1_b64,
        heatmap2=heatmap2_b64,
        img_minucias1=img_minucias1_b64,
        img_minucias2=img_minucias2_b64)

    except Exception as e:
        logging.error("Match processing error [req=%s]", req_id, exc_info=True)
        return jsonify({'error': 'Erro interno no processamento. Tente novamente.'}), 500
    finally:
        for _p in (path1, path2):
            try:
                if os.path.exists(_p):
                    os.remove(_p)
            except OSError:
                pass

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
