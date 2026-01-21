# Multiclass Disease Classification using Blood Sample Data

## a. Problem Statement

The objective of this project is to design, implement, and deploy a **multiclass machine learning classification system** that predicts the **disease category of a patient** based on blood sample parameters.

Given a dataset containing multiple **numerical blood test features**, each patient record must be classified into **one disease category out of multiple possible disease classes**. This problem is formulated as a **supervised multiclass classification task**, where each instance belongs to exactly one target class.

The project aims to:
- Implement and compare multiple machine learning classification models on the same dataset  
- Evaluate each model using standard performance metrics such as Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)  
- Analyze and compare the performance of individual and ensemble models  
- Deploy the trained models through an interactive **Streamlit web application** that allows dataset upload, model selection, evaluation, and visualization  

This demonstrates a complete **end-to-end machine learning workflow**, covering data preprocessing, model training, evaluation, comparative analysis, and real-world deployment.


---

## b. Dataset Description

- **Dataset Name:** Multiple Disease Prediction based on Blood Samples  
- **Source:** Kaggle - https://www.kaggle.com/datasets/ehababoelnaga/multiple-disease-prediction/data
- **Total Samples:** 2,837
- **Total Columns:** 25
  - 24 Feature columns
  - 1 Target column
- **Problem Type:** Multiclass Classification

Each row in the dataset represents a single patient’s blood test record.  
The objective is to classify each record into **one disease category** based on blood sample parameters.

---

## Feature Columns

The dataset contains the following **24 numerical blood test features**:

1. Glucose  
2. Cholesterol  
3. Hemoglobin  
4. Platelets  
5. White Blood Cells  
6. Red Blood Cells  
7. Hematocrit  
8. Mean Corpuscular Volume  
9. Mean Corpuscular Hemoglobin  
10. Mean Corpuscular Hemoglobin Concentration  
11. Red Cell Distribution Width  
12. Serum Iron  
13. Total Iron Binding Capacity  
14. Ferritin  
15. Creatinine  
16. Blood Urea Nitrogen  
17. Troponin  
18. C-reactive Protein  
19. Alanine Aminotransferase  
20. Aspartate Aminotransferase  
21. Alkaline Phosphatase  
22. Albumin  
23. Globulin  
24. Bilirubin  

All feature columns are **numerical**.

---

## Target Column            

### Target Column Name
- **`Disease`**

### Target Class Labels

The target variable is a **multiclass categorical label** with the following **six disease categories**:

- **Anemia**
- **Diabetes**
- **Healthy**
- **Heart Di** *(Heart Disease)*
- **Thalasse** *(Thalassemia)*
- **Thromboc** *(Thrombocytopenia)*

---

## c. Models Used - Comparison Table with the evaluation metrics

| ML Model Name | Accuracy | AUC | Precision | Recall |   F1   | MCC |
|---------------|----------|-----|-----------|--------|--------|-----|
| Logistic Regression | 0.8820 | 0.9209 | 0.8765 | 0.8820 | 0.8779 | 0.8476 |
| Decision Tree | 0.9683 | 0.9474 | 0.9678 | 0.9683 | 0.9673 | 0.9590 |
| KNN | 0.9137 | 0.9277 | 0.9184 | 0.9137 | 0.9090 | 0.8898 |
| Naive Bayes | 0.8151 | 0.9515 | 0.8139 | 0.8151 | 0.8120 | 0.7593 |
| Random Forest (Ensemble) | 0.9648 | 0.9981 | 0.9538 | 0.9648 | 0.9584 | 0.9549 |
| XGBoost (Ensemble) | 0.9859 | 0.9975 | 0.9833 | 0.9859 | 0.9832 | 0.9818 |

---

## Observation about model performance

| ML Model Name      | Observation about model performance                                                                                                                    |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression| Linear model; moderately accurate (0.8820) and MCC (0.8476). Can struggle to capture complex decision boundaries, explaining why it trails non-linear approaches. |
| Decision Tree      | High accuracy (0.9683) and MCC (0.9590). Learns non-linear relationships well but may slightly underperform in generalization compared to ensembles due to risk of overfitting. |
| KNN                | Solid accuracy (0.9137) and MCC (0.8898); a non-parametric method sensitive to feature scaling and curse of dimensionality, which limits its advantage over tree-based methods. |
| Naive Bayes        | Fastest due to simple computations, but lowest accuracy (0.8151) and MCC (0.7593). Assumes feature independence, which rarely holds in practice; notable for highest AUC (0.9515) showing good ranking ability on this dataset. |
| Random Forest (Ensemble)      | Combines many decision trees to improve generalization; excellent accuracy (0.9648) and MCC (0.9549), reducing variance and overfitting but slightly less performant than XGBoost. |
| XGBoost (Ensemble)            | Gradient boosting ensemble. Best overall metrics: accuracy (0.9859) and MCC (0.9818); excels due to regularization, weighted learning, and handling of complex feature interactions—sets the benchmark for this dataset.   |
---

