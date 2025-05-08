import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
from cam_utils import generate_gradcam_heatmap, overlay_heatmap_on_image

app = Flask(__name__)
MODEL_PATH = 'model/siamese_model.h5'
THRESHOLD = 15.5  # 🔄 Novo limiar mais seletivo

# Função de distância Euclidiana usada na camada Lambda
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

# Carregando o modelo com a função customizada
model = load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={'euclidean_distance': euclidean_distance}
)

# Pré-processamento da imagem
def preprocess(img_file):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_bmp(img, channels=3)
    img = tf.image.resize(img, [96, 96])
    img = tf.cast(img, tf.float32) / 255.0
    return img

@app.route('/')
def index():
    return send_from_directory('.', 'upload.html')

@app.route('/match', methods=['POST'])
def match():
    try:
        img1 = request.files['img1']
        img2 = request.files['img2']
        show_heatmap = 'heatmap' in request.form

        path1 = 'temp1.bmp'
        path2 = 'temp2.bmp'
        img1.save(path1)
        img2.save(path2)
        print("🖼️  Imagens recebidas e salvas:", path1, path2)

        # Pré-processamento
        input1 = preprocess(path1)
        input2 = preprocess(path2)
        input1 = tf.expand_dims(input1, axis=0)
        input2 = tf.expand_dims(input2, axis=0)
        print("🧪 Pré-processamento concluído")

        # Inferência
        prediction = model.predict([input1, input2])[0][0]
        match = prediction < THRESHOLD
        resultado_texto = 'MATCH' if match else 'NO MATCH'
        print(f"✅ Score: {prediction:.4f} | Limiar: {THRESHOLD} | Resultado: {resultado_texto}")

        # Geração do mapa de calor (opcional)
        heatmap1_b64 = None
        heatmap2_b64 = None
        if show_heatmap:
            heatmap1 = generate_gradcam_heatmap(model, input1)
            heatmap2 = generate_gradcam_heatmap(model, input2)
            heatmap1_b64 = overlay_heatmap_on_image(path1, heatmap1)
            heatmap2_b64 = overlay_heatmap_on_image(path2, heatmap2)

        return render_template_string('''
        <!DOCTYPE html>
        <html lang="pt-br">
        <head>
          <meta charset="UTF-8">
          <title>Resultado</title>
          <style>
            body {
              font-family: 'Segoe UI', sans-serif;
              background: #f5f6fa;
              margin: 0;
              padding: 0;
            }
            .result-container {
              background: white;
              max-width: 640px;
              margin: 60px auto;
              padding: 30px;
              border-radius: 12px;
              text-align: center;
              box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            h1 {
              color: #2f3640;
              margin-bottom: 20px;
            }
            p {
              font-size: 16px;
              margin: 10px 0;
            }
            .match { color: green; font-weight: bold; font-size: 18px; }
            .nomatch { color: red; font-weight: bold; font-size: 18px; }
            img {
              width: 300px;
              margin: 15px;
              border: 1px solid #ccc;
              border-radius: 6px;
            }
            .btn {
              display: inline-block;
              margin-top: 20px;
              background: #0984e3;
              color: white;
              padding: 10px 18px;
              border-radius: 6px;
              text-decoration: none;
              font-weight: bold;
            }
            .btn:hover {
              background: #74b9ff;
            }
          </style>
        </head>
        <body>
          <div class="result-container">
            <h1>Resultado da Comparação</h1>
            <p><strong>Score:</strong> {{ score }}</p>
            <p class="{{ 'match' if resultado == 'MATCH' else 'nomatch' }}">Resultado: {{ resultado }}</p>

            {% if heatmap1 %}
              <h3>Mapa de Calor - Imagem 1</h3>
              <img src="data:image/png;base64,{{ heatmap1 }}">
              <h3>Mapa de Calor - Imagem 2</h3>
              <img src="data:image/png;base64,{{ heatmap2 }}">
            {% endif %}

            <a class="btn" href="/">↩ Voltar</a>
          </div>
        </body>
        </html>
        ''',
        score=round(float(prediction), 4),
        resultado=resultado_texto,
        heatmap1=heatmap1_b64,
        heatmap2=heatmap2_b64)

    except Exception as e:
        print("❌ Erro durante comparação:")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 🔍 Diagnóstico: listar camadas internas da sub-rede MobileNetV2
def listar_camadas_mobilenet():
    try:
        submodelo = model.get_layer('functional')
        print("📋 Camadas da sub-rede MobileNetV2 dentro do modelo siamesa:")
        for layer in submodelo.layers:
            print(" -", layer.name)
    except Exception as e:
        print("❌ Erro ao acessar sub-rede MobileNetV2:")
        import traceback
        traceback.print_exc()

listar_camadas_mobilenet()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
 
