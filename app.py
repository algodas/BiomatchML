
import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
from cam_utils import generate_gradcam_heatmap, overlay_heatmap_on_image
import base64
import cv2
from skimage.morphology import skeletonize
from skimage.util import invert

app = Flask(__name__)
MODEL_PATH = 'model/siamese_model.h5'

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

model = load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={'euclidean_distance': euclidean_distance}
)

def preprocess(img_file):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_bmp(img, channels=3)
    img = tf.image.resize(img, [96, 96])
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

        threshold = 15.5 if mode == '1to1' else 4.0
        alert_min, alert_max = (14.3, 15.8) if mode == '1to1' else (3.5, 4.5)

        input1 = preprocess(path1)
        input1 = tf.expand_dims(input1, axis=0)

        if mode == '1to1':
            img2 = request.files['img2']
            path2 = 'temp2.bmp'
            img2.save(path2)
            input2 = preprocess(path2)
            input2 = tf.expand_dims(input2, axis=0)
            prediction = model.predict([input1, input2])[0][0]
            result_text = 'MATCH' if prediction < threshold else 'NO MATCH'
            if alert_min <= prediction <= alert_max:
                result_text += '\n⚠️ For this comparison, a forensic expert review is recommended.'
            minucias2 = detectar_minucias(path2)
            img_minucias2_b64 = gerar_imagem_com_pontos(path2, minucias2)
            if show_heatmap:
                heatmap2 = generate_gradcam_heatmap(model, input2)
                heatmap2_b64 = overlay_heatmap_on_image(path2, heatmap2)
            else:
                heatmap2_b64 = None

        else:  # 1:N
            embeddings = np.load('database/templates.npy')
            with open('database/filenames.txt') as f:
                filenames = [line.strip() for line in f]
            query_embedding = model.get_layer('functional').predict(input1)[0]
            distances = np.linalg.norm(embeddings - query_embedding, axis=1)
            idx = np.argmin(distances)
            prediction = distances[idx]
            candidate_path = os.path.join('database/imagens', filenames[idx])
            result_text = f"MATCH with: {filenames[idx]}" if prediction < threshold else f"NO MATCH. Closest match: {filenames[idx]}"
            if alert_min <= prediction <= alert_max:
                result_text += '\n⚠️ For this comparison, a forensic expert review is recommended.'
            minucias2 = detectar_minucias(candidate_path)
            img_minucias2_b64 = gerar_imagem_com_pontos(candidate_path, minucias2)
            if show_heatmap:
                candidate_input = preprocess(candidate_path)
                candidate_input = tf.expand_dims(candidate_input, axis=0)
                heatmap2 = generate_gradcam_heatmap(model, candidate_input)
                heatmap2_b64 = overlay_heatmap_on_image(candidate_path, heatmap2)
            else:
                heatmap2_b64 = None

        minucias1 = detectar_minucias(path1)
        img_minucias1_b64 = gerar_imagem_com_pontos(path1, minucias1)
        heatmap1_b64 = None
        if show_heatmap:
            heatmap1 = generate_gradcam_heatmap(model, input1)
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
    <p class='{{ 'match' if resultado.startswith('MATCH') else 'nomatch' }}'>Result: {{ resultado }}</p>
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
 
