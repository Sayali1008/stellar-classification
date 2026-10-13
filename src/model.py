from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import FOREST_PARAMS, RANDOM_STATE, TARGET, TEST_SIZE
from src.features import add_colors, select_final_features


def split_data(df, features):
    """Split: 80% train, 20% test, same class proportions in both."""
    X = df[features]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def build_model(**forest_params):
    """Raw u, g, r, i, z in → colors added → final features selected → random forest.

    Uses FOREST_PARAMS from config unless other forest settings are passed in (for comparing sizes).
    """
    params = forest_params or FOREST_PARAMS
    return Pipeline([
        ("colors", FunctionTransformer(add_colors)),
        ("features", FunctionTransformer(select_final_features)),
        ("model", RandomForestClassifier(**params, n_jobs=-1, random_state=RANDOM_STATE)),
    ])


def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
