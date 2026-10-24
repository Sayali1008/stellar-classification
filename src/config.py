from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

# Saved model + its metadata, written by train.py and loaded by the API
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'model.joblib'
METADATA_PATH = ARTIFACTS_DIR / 'metadata.json'

# ID columns are loaded as strings: they label things, they are not measurements
ID_COLS = ['obj_ID', 'spec_obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'plate', 'fiber_ID']

FILTERS = ['u', 'g', 'r', 'i', 'z']
COLORS = ['u_g', 'g_r', 'r_i', 'i_z']

# Final feature set: 4 colors + r (same information as all five filters, chosen by cross-validation)
FINAL_FEATURES = COLORS + ['r']

TARGET = 'class'
CLASSES = ['GALAXY', 'STAR', 'QSO']

# -9999 is a placeholder for no measurement
PLACEHOLDER = -9999

# Training-data checks (src/validate.py). The real data has ~100,000 rows and magnitudes between
# about 9.5 and 33, so these bounds catch broken uploads and garbage values, not faint real objects.
MIN_ROWS = 10_000
# Largest share of rows allowed to hold the -9999 placeholder before cleaning drops them.
# Today it is 1 row in 100,000; a broken export with many placeholders must fail, not shrink quietly.
MAX_PLACEHOLDER_SHARE = 0.01
MAG_MIN = 5
MAG_MAX = 40

# Out-of-range warnings (src/predict.py): the training range of each filter, recorded in metadata.json,
# runs from the 1st to the 99th percentile. About 6% of real objects fall outside it on some filter.
INPUT_RANGE_QUANTILES = (0.01, 0.99)

# Most objects accepted in one /predict request (keeps requests under Lambda's 6 MB payload limit)
MAX_BATCH_SIZE = 1000

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Forest size, chosen by 5-fold CV on the training split (scripts/compare_forest_sizes.py):
# the smallest setting within one standard deviation of the best macro F1.
# Best: 100 trees, min_samples_leaf=5, CV 0.8545 ± 0.0021 (15 MB). Chosen: 0.8531 at 7.5 MB,
# versus 0.8544 at 48 MB for the original 200 unrestricted trees.
FOREST_PARAMS = {'n_estimators': 50, 'min_samples_leaf': 5}

# AWS S3 data location
DATA_BUCKET = 'amzn-s3-stellar-data'
DATA_KEY = 'raw/star_classification.csv'

# Every model that passes the deploy gate is stored at s3://DATA_BUCKET/MODELS_PREFIX/<model_version>/
MODELS_PREFIX = 'models'

# The live API. The deploy gate asks its /health which model is serving, to compare against it.
LIVE_URL = 'https://7ld2mronnk2hmj5mfmdup3h4rq0lxwbj.lambda-url.us-east-2.on.aws'

# Deploy gate (src/gate.py), on the new test split:
# a new model must reach the floor, and score at least the live model's macro F1 minus the tolerance.
# The tolerance lets a model that ties (same data, same seed) or differs by noise through.
GATE_MIN_F1 = 0.84
F1_TOLERANCE = 0.005

# MLflow: params and metrics go to mlflow.db at the project root
# run files (plots, models) go to the private S3 bucket, next to the training data.
# Absolute path, so every script uses the same mlflow.db whichever folder it runs from.
MLFLOW_TRACKING_URI = f'sqlite:///{PROJECT_ROOT / "mlflow.db"}'
MLFLOW_ARTIFACT_ROOT = f's3://{DATA_BUCKET}/mlflow'