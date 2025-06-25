
# 🫁 Lung Cancer Classification using Machine Learning

This project aims to classify lung cancer presence using a combination of structured data and machine learning models. The notebook includes preprocessing, model training, evaluation, and analysis.

---

## 📌 Project Description

The goal is to detect lung cancer based on a structured dataset by training several machine learning classifiers. The dataset contains clinical and demographic information. The models explored include:

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost
- Logistic Regression
- Gradient Boosting

---

## 📈 Results Summary

| Model               | Accuracy   |
|--------------------|------------|
| Logistic Regression| 77.8966%   |
| KNN                | 73.7438%   |
| Decision Tree      | 64.0397%   |
| Random Forest      | 77.8854%   |
| XGBoost            | 77.9775%   |
| Gradient Boosting  | 77.9691%   |

---

## ❌ Why Accuracy Couldn't Reach the Best Optimum

Despite applying common preprocessing steps such as feature engineering, outlier removal, and normalization, the accuracy didn’t reach the highest possible benchmarks due to the following limitations:

1. **Hardware Constraints**  
   - GPU: GTX 1650 4GB  
   - RAM: 16GB DDR3  
   - CPU: AMD Ryzen 5 3550U  
   These specifications limit the ability to perform heavy hyperparameter tuning, deep learning experimentation, or training on large ensembles.

2. **Dataset Limitations**  
   - Limited dataset size and diversity  
   - Potential class imbalance and missing domain-specific features  

3. **Model Complexity Trade-offs**  
   - Unable to use deeper neural networks or transformer-based models due to RAM and GPU limits.

---

## 🔭 Scope for Future Improvements

Here’s how this project can be improved further with better resources and methods:

1. **Use of Deep Learning Models**  
   Incorporating CNNs or hybrid architectures if imaging data is included.

2. **Hyperparameter Optimization**  
   Using tools like Optuna, GridSearchCV with parallelism for better model tuning.

3. **Feature Selection & Dimensionality Reduction**  
   PCA, Recursive Feature Elimination (RFE), and mutual information analysis.

4. **Class Imbalance Handling**  
   Apply SMOTE, ADASYN, or ensemble methods with cost-sensitive learning.

5. **Deployment**  
   Deploy as a web app using Flask/Streamlit for real-world accessibility.

6. **Hardware Upgrade**  
   With a better GPU (RTX 3060 or above) and DDR4/DDR5 RAM, model training speed and accuracy can significantly improve.

---

## 🧠 Technologies Used

- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- Jupyter Notebook
- Matplotlib / Seaborn
- Machine Learning Classification Algorithms (including Gradient Boosting)

---