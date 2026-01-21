import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize

# ---------------- Page Config ----------------
st.set_page_config(page_title="Disease Classification using Blood Sample Data", layout="wide")

TARGET_COL = "Disease"

st.title("Multiclass Disease Classification using Blood Sample Data")
st.write("Try and Compare Performance Across Different ML Models")

# ---------------- Sidebar ----------------
st.sidebar.header("Model Selection")

model_dict = {
    "Logistic Regression": "model_pkl/logistic.pkl",
    "Decision Tree": "model_pkl/decision_tree.pkl",
    "KNN": "model_pkl/knn.pkl",
    "Naive Bayes": "model_pkl/naive_bayes.pkl",
    "Random Forest": "model_pkl/random_forest.pkl",
    "XGBoost": "model_pkl/xgboost.pkl"
}

model_name = st.sidebar.selectbox("Choose a model", list(model_dict.keys()))
model = joblib.load(model_dict[model_name])

# ------------------------------------------------
# Learn CLASS_NAMES ONLY from LOCAL data/test.csv
# ------------------------------------------------
df_local_test = pd.read_csv("data/test.csv")

if TARGET_COL not in df_local_test.columns:
    st.error("Local data/test.csv must contain 'Disease' column to infer class names.")
    st.stop()

CLASS_NAMES = sorted(df_local_test[TARGET_COL].unique())

# ---------------- Download test.csv ----------------
st.sidebar.subheader("Sample Test Data")

with open("data/test.csv", "rb") as f:
    st.sidebar.download_button(
        label="Download test.csv",
        data=f,
        file_name="test.csv",
        mime="text/csv"
    )

# ---------------- Upload CSV ----------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Inform user of evaluation vs prediction-only modes
st.info(
    "This App can work in two modes:\n\n"
    "- **Evaluation mode:** If your CSV contains a 'Disease' (target) column, the model will assess its performance (compute metrics and confusion matrix).\n"
    "- **Prediction-only mode:** If your CSV does not contain the target column, the model will output predictions only.\n"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file for prediction/evaluation.")
    st.stop()

required_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
missing_cols = []
if required_features is not None:
    missing_cols = [col for col in required_features if col not in df.columns]
if missing_cols:
    st.error(
        f"Uploaded CSV is missing required columns: {', '.join(missing_cols)}.\n"
        "Please ensure your file matches the model input format."
    )
    st.stop()

# ---------------- Mode Detection (ONLY on uploaded CSV) ----------------
evaluation_mode = TARGET_COL in df.columns

if evaluation_mode:
    st.info("Evaluation mode: Target column detected in uploaded file")

    X_test = df.drop(columns=[TARGET_COL])

    # Map string labels → indices using CLASS_NAMES from local test.csv
    try:
        y_test = df[TARGET_COL].apply(lambda x: CLASS_NAMES.index(x)).values
    except ValueError:
        st.error("Uploaded CSV contains unseen class labels.")
        st.stop()
else:
    st.info("Prediction-only mode: No target column in uploaded file")

    X_test = df
    y_test = None

# ---------------- Prediction ----------------
y_pred = model.predict(X_test)

# Decode predictions using CLASS_NAMES
y_pred_labels = [CLASS_NAMES[i] for i in y_pred]

# ---------------- Display Predictions ----------------
if not evaluation_mode:
    st.subheader("Predictions")
    # Show all uploaded columns + predicted disease
    display_df = df.copy()
    display_df["Predicted Disease"] = y_pred_labels
    st.dataframe(display_df)

# ---------------- Metrics & Confusion Matrix ----------------
if evaluation_mode:
    y_prob = model.predict_proba(X_test)

    num_classes = len(CLASS_NAMES)
    y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))

    # Compute metrics first
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    # Arrange metrics in the requested order for display
    metrics_ordered = [
        ("Accuracy", accuracy),
        ("AUC Score", auc),
        ("Precision", precision),
        ("Recall", recall),
        ("F1 Score", f1),
        ("Matthews Correlation Coefficient (MCC Score)", mcc)
    ]

    st.subheader("Evaluation Metrics")
    st.table(pd.DataFrame(metrics_ordered, columns=["Metric", "Value"]))


    # ---------------- Confusion Matrix ----------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    # ↓↓↓ Reduce figure size from (8, 6) to (6, 4.5)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
