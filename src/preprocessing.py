from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def build_preprocessor(numeric_features: list,
                         categorical_features: list,
                         tree: bool = False):

      if tree:
          numeric_transformer = "passthrough"
      else:
          numeric_transformer = Pipeline([
              ("scaler", StandardScaler()),
              ("poly", PolynomialFeatures(include_bias=False))
          ])

      categorical_transformer = Pipeline([
          ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
      ])

      preprocessor = ColumnTransformer([
          ("num", numeric_transformer, numeric_features),
          ("cat", categorical_transformer, categorical_features)
      ])
      return preprocessor



def get_train_test(df: DataFrame,
                  target_col: str,
                  to_drop: list|None = None):

    if to_drop:
        df = df.drop(columns=to_drop)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return Xtrain, Xtest, ytrain, ytest


