
# 🧬 Fingerprint Matching System (Biometric Siamese Network + Explainability)

This project implements a fingerprint comparison system using a **Siamese Neural Network**, enhanced with:
- ✅ Cosine or Euclidean similarity scoring
- ✅ Grad-CAM heatmaps for explainability
- ✅ Automatic **minutiae (keypoint) detection** on each fingerprint

---

## 🚀 Features

- 📷 Upload two fingerprint images (`.bmp`)
- 🔍 Calculates a **similarity score** using a deep learning model
- 🔥 **Grad-CAM heatmaps** to highlight regions that influenced the decision
- 🧬 **Keypoint (minutiae) visualization** — automatically detected and rendered on top of each image
- ⚠️ Caution message if score falls within a "gray zone" requiring forensic expert review
- 🧪 Lightweight backend with Flask and TensorFlow
- 🌐 User interface with HTML/CSS (single-page form)

---

## 🖼️ Example Output

After matching, the system presents:

- Similarity score (e.g. `12.45`)
- Match/No Match decision
- Optional Grad-CAM heatmaps (if checkbox selected)
- Automatic keypoint visualization (minutiae detection)

---

## 🛠 Requirements

```bash
pip install -r requirements.txt
```

You will need:
- TensorFlow
- Flask
- OpenCV
- scikit-image
- NumPy

---

## 📁 File Structure

- `app.py` — Main Flask server
- `upload.html` — Web interface
- `model/siamese_model.h5` — Pretrained fingerprint matching model
- `cam_utils.py` — Grad-CAM generation and overlay tools
- `requirements.txt` — Python dependencies

---

## 📡 How to Run

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧪 Future Ideas

- Auto-tuning of threshold based on feedback
- One-to-many fingerprint search
- Finger classification (thumb, index, etc.)
- Fingerprint spoof/liveness detection

---

## 👤 Author

Developed by [@algodas](https://www.linkedin.com/in/algodas/) | Contact for collaborations or suggestions!
