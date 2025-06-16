# 🎓 Student Performance Prediction

Predict student exam performance using demographic, academic, and behavioral data. This end‑to‑end ML pipeline aids early interventions and targeted academic support.

---

## 📁 Project Structure

```
student_performance_prediction/
├── data/
│   └── student_data.csv               # Raw input dataset
├── notebooks/
│   ├── 1_data_exploration.ipynb       # EDA: distributions, correlations, outliers
│   ├── 2_preprocessing.ipynb         # Cleaning, encoding, scaling
│   └── 3_modeling.ipynb              # Training, evaluation, feature analysis
├── src/
│   ├── data_loader.py                # Load & split dataset
│   ├── preprocessing.py             # Pipelines for feature transformation
│   ├── train_model.py               # Train ML models and tune hyper‑parameters
│   ├── evaluate_model.py            # Evaluate and visualize model performance
│   └── predict.py                   # Infer new student performance
├── artifacts/                        # Saved models & preprocessors
├── requirements.txt                 # Project dependencies
└── README.md                        # This file
```

---

## 🧠 Overview

- **Problem**: Predict numeric grades or pass/fail outcomes based on features like gender, study time, attendance, previous scores, and more.
- **Approach**:
  - **EDA**: Visualize relationships and target predictors (e.g., study hours vs. scores).
  - **Preprocessing**: Clean data, impute missing values, encode categoricals, scale numerics.
  - **Modeling**: Train and compare models—Linear Regression, Random Forest, XGBoost, SVM.
  - **Evaluation**: Use RMSE for regression; accuracy, F1-score, ROC-AUC for classification.
  - **Interpretation**: Evaluate feature importance and key model drivers.

---

## ⚙️ Installation

1. **Clone repo**:
   ```bash
   git clone https://github.com/sujalk777/ML_Projects.git
   cd ML_Projects/student_performance_prediction
   ```

2. **Set up environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Run notebooks (EDA, preprocessing, modeling)
```bash
jupyter notebook notebooks/1_data_exploration.ipynb
```

### Train & save best model
```bash
python src/train_model.py \
  --data data/student_data.csv \
  --output_dir artifacts/
```

### Evaluate model
```bash
python src/evaluate_model.py \
  --data data/student_data.csv \
  --model artifacts/best_model.pkl
```

### Inference on new data
```bash
python src/predict.py \
  --input_json '{
      "gender":"female",
      "ethnicity":"group B",
      "studytime":3,
      "failures":0,
      "attendance":95,
      "previous_score":78
  }' \
  --model artifacts/best_model.pkl \
  --preprocessor artifacts/preprocessor.pkl
```

---

## 🏆 Results

- **Regression**: RMSE = Y, R² = Z
- **Classification**: Accuracy = X% (pass/fail)
- **Key predictors**:
  - Study time, attendance, past performance, parental education

---

## 🔧 Future Work

- Add features: extracurriculars, tutoring, socioeconomic factors
- Test advanced models: CatBoost, neural networks
- Build a web interface (Flask/Streamlit)
- Deploy with monitoring and CI/CD pipeline

---

## 📦 Requirements

- Python 3.7+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`

Refer to **requirements.txt** for detailed dependencies.

---

## 🤝 Contributing

Enhancements are welcome! To contribute:
1. Fork the repository  
2. Create a branch: `git checkout -b feature/your_feature`
3. Commit changes: `git commit -m "Add feature"`
4. Push: `git push origin feature/your_feature`
5. Open a Pull Request

---

## 📄 License

MIT © [Your Name]. See the LICENSE file for details.
