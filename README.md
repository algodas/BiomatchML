# 🧬 Fingerprint Matching System (Biometric Siamese Network + Explainability)

This project implements a fingerprint comparison system using a Siamese Neural Network, enhanced with:

✅ Cosine or Euclidean similarity scoring  
✅ Grad-CAM heatmaps for explainability  
✅ Automatic minutiae (keypoint) detection on each fingerprint  

---

## 🚀 Features

- 📷 Upload two fingerprint images (.bmp or .png)  
- 🔍 Calculates a similarity score using a deep learning model  
- 🔥 Grad-CAM heatmaps to highlight regions that influenced the decision  
- 🧬 Keypoint (minutiae) visualization — automatically detected and rendered on top of each image  
- ⚠️ Caution message if score falls within a "gray zone" requiring forensic expert review  

---

## 🧪 Lightweight Architecture

- **Backend**: Flask + TensorFlow  
- **Frontend**: HTML/CSS (single-page form)  

---

## 🌐 Live Demo

Test the application online:  
👉 [https://projetos.tiago.cafe/](https://projetos.tiago.cafe/)

---

## 🖼️ Example Output

After matching, the system presents:

- ✅ Similarity score (e.g. `12.45`)  
- ✅ Match / No Match decision  
- ✅ Optional Grad-CAM heatmaps  
- ✅ Automatic keypoint visualization (minutiae detection)  

---

## 🛠 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt

