
import pandas as pd, joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor(X):
    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object", "category"]).columns
    return ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])


df = pd.read_csv("data/train.csv")
X, y = df.drop("Disease", axis=1), df["Disease"]

pipe = Pipeline([
    ("preprocessing", get_preprocessor(X)),
    ("classifier", XGBClassifier(
        objective="multi:softprob",
        num_class=y.nunique(),
        eval_metric="mlogloss"
    ))
])

pipe.fit(X, y)
joblib.dump(pipe, "model_pkl/xgboost.pkl")
print("XGBoost model saved to model_pkl/xgboost.pkl")