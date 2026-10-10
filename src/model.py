from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import N_ESTIMATORS, RANDOM_STATE, TARGET, TEST_SIZE


def split_data(df, features):
    """Split: 80% train, 20% test, same class proportions in both."""
    X = df[features]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def build_model():
    return RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=RANDOM_STATE)


def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
