# 🌾 Crop Disease Prediction Web App

An AI-powered crop disease detection system using 
Transfer Learning (MobileNet-V2) and LLM-generated 
disease explanations.

## 🎯 Key Results
- **97.9% accuracy** on test data
- Precision: 0.99 | Recall: 0.98 (Virus class)
- Trained on 525 images across 2 crop disease classes

## 🛠️ Tech Stack
- **ML Model:** TensorFlow/Keras — MobileNet-V2 (Transfer Learning)
- **Backend:** Flask (Python)
- **AI Explanations:** Groq LLM API
- **Data Processing:** NumPy, Pandas, Pillow
- **Frontend:** HTML, CSS, JavaScript

## 📊 Model Performance
![Accuracy](fig3_accuracy.png)
![Confusion Matrix](fig6_confusion_matrix.png)

## 🚀 How to Run
```bash
pip install -r requirements.txt
cp .env.example .env   # Add your Groq API key
python app.py
```
Visit: http://127.0.0.1:5000

## 📁 Project Structure
- `app.py` — Flask web server
- `basic_model.py` — Model training script
- `compare_models.py` — Model comparison experiments
- `explain.py` — Groq LLM integration
- `crop_model.pkl` — Saved trained model
- `Crop_recommendation.csv` — Supplement recommendations
