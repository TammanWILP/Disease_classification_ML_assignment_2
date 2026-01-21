
import pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
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
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipe.fit(X, y)
joblib.dump(pipe, "model_pkl/random_forest.pkl")
print("random_forest model saved to model_pkl/random_forest.pkl")