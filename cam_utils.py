import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import io
import base64


def generate_gradcam_heatmap(model, img_array, layer_name='conv2d_1'):
    """
    Gera o mapa de calor Grad-CAM para uma imagem de entrada e uma camada específica.
    :param model: modelo siamesa com submodelo MobileNetV2 customizado
    :param img_array: imagem (96x96x3) numpy array normalizada [0-1]
    :param layer_name: nome da camada final convolucional do submodelo
    :return: heatmap como numpy array
    """
    # Acessa o submodelo funcional dentro do modelo siamesa
    mobilenet_submodel = model.get_layer('functional')

    # Gera modelo auxiliar com saída na camada de interesse e entrada no submodelo
    grad_model = tf.keras.models.Model(
        [mobilenet_submodel.input],
        [mobilenet_submodel.get_layer(layer_name).output]
    )

    with tf.GradientTape() as tape:
        conv_outputs = grad_model(img_array)
        loss = tf.reduce_mean(conv_outputs)

    # Gradiente da saída em relação à feature map
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # Multiplica cada canal pela sua importância
    conv_outputs = conv_outputs[0].numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Gera heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10

    return heatmap


def overlay_heatmap_on_image(img_path, heatmap, alpha=0.4):
    """
    Sobrepõe o heatmap na imagem original.
    :param img_path: caminho da imagem original
    :param heatmap: mapa de calor como array numpy
    :param alpha: transparência
    :return: imagem combinada como base64
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (96, 96))
    heatmap = cv2.resize(heatmap, (96, 96))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)

    # Converte para base64 para web
    _, buffer = cv2.imencode('.png', superimposed_img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image


# Exemplo de uso no app:
# heatmap = generate_gradcam_heatmap(model, preprocessed_img_tensor)
# heatmap_img = overlay_heatmap_on_image("temp1.bmp", heatmap)
# --> inserir heatmap_img na resposta HTML com: <img src="data:image/png;base64,{heatmap_img}">
 
