from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import CLASSES, FINAL_FEATURES, FOREST_PARAMS, RANDOM_STATE, TARGET, TEST_SIZE
from src.features import add_colors, select_final_features


def split_data(df, features):
    """Split: 80% train, 20% test, same class proportions in both."""
    X = df[features]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def build_model(features=FINAL_FEATURES, **forest_params):
    """Raw u, g, r, i, z in → colors added → chosen features selected → random forest.

    Uses FINAL_FEATURES and FOREST_PARAMS from config unless others are passed in (for comparisons).
    Forest settings replace FOREST_PARAMS as a whole, they are not merged into it.
    """
    params = forest_params or FOREST_PARAMS
    return Pipeline([
        ("colors", FunctionTransformer(add_colors)),
        ("features", FunctionTransformer(select_final_features, kw_args={'features': features})),
        ("model", RandomForestClassifier(**params, n_jobs=-1, random_state=RANDOM_STATE)),
    ])


def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def cv_scores(model, X_train, y_train):
    """5-fold cross-validation on the training split: macro F1 and each class's recall per fold.

    Same folds on every call (stratified, so each fold keeps the class mix), so candidates are
    compared on the same objects. Returns {'macro_f1': array of 5, 'recall_STAR': array of 5, ...}.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {'macro_f1': 'f1_macro'}
    for c in CLASSES:
        # labels=[c] scores just this class's recall
        scoring[f'recall_{c}'] = make_scorer(recall_score, labels=[c], average='macro')
    results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    return {name: results[f'test_{name}'] for name in scoring}
