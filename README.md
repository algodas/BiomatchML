🧬 Biometric Comparator with Siamese Neural Network
This project implements a fingerprint comparator using a Siamese neural network with support for Grad-CAM visualizations.

⚙️ Features
BMP image comparison

Web interface with upload and Grad-CAM support

Flask API deployed with Gunicorn + NGINX

Automatic training via train.py

🚀 Automatic Installation
bash
Copiar
Editar
chmod +x setup.sh
./setup.sh
🧪 Dataset Used
Use the SOCOFing Dataset or your own images placed in the training directory.

📁 Project Structure
app.py: Main Flask API

train.py: Training script for the Siamese model

cam_utils.py: Grad-CAM heatmap generation

templates/upload.html: Web interface

model/: Folder where the .h5 model is saved

demo_images/: Sample images

deploy/: System configuration files
