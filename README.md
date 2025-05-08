
# 🧬 Fingerprint Comparator with Siamese Network

This project implements a biometric fingerprint comparator using a lightweight Siamese Neural Network (CNN) with a modern web interface. It's designed for use in environments with limited hardware (CPU-only) and provides real-time comparison as well as visual heatmap explanations via Grad-CAM.

## 🚀 Features

- Upload and compare two fingerprint BMP images
- Displays similarity score and match decision (`MATCH` or `NO MATCH`)
- Optional Grad-CAM heatmaps showing areas of most attention
- Stylish, mobile-friendly frontend
- Lightweight model deployable without GPU

## 🧪 Dataset Used

- **SOCOFing Dataset**: 6000+ real and altered fingerprint images in BMP format
- Includes variation by rotation, noise, cuts, and partial captures

## 📦 Project Structure

```
fingerprint-app/
├── app.py                  # Flask API with web interface
├── cam_utils.py            # Grad-CAM visualization module
├── train.py                # Model training script (CPU friendly)
├── model/                  # Folder for the trained model (.h5)
├── templates/upload.html   # Web interface (styled)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── deploy/                 # Deployment configs (NGINX + Gunicorn)
```

## 📥 Installation

```bash
git clone https://github.com/yourname/fingerprint-app.git
cd fingerprint-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🧠 Training

```bash
python train.py
# The trained model will be saved at model/siamese_model.h5
```

## ▶️ Run Locally

```bash
python app.py
# Visit http://localhost:5000
```

## 🔧 Deployment with Gunicorn + NGINX

- Copy `deploy/gunicorn.service` to `/etc/systemd/system/`
- Start and enable:
```bash
sudo systemctl start gunicorn
sudo systemctl enable gunicorn
```

- Copy `deploy/nginx.conf` to `/etc/nginx/sites-available/fingerprint`
- Enable site and restart NGINX:
```bash
sudo ln -s /etc/nginx/sites-available/fingerprint /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

## 🧊 License

This project is for educational and exploratory use.
